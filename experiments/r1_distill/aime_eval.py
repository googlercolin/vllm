import pandas as pd
import csv
import os
import shutil
import requests
import transformers
from tqdm import tqdm
import random
import re
import argparse
from datasets import load_dataset

def setup_directories():
    """Set up necessary directories and return path constants."""
    HOME = os.path.expanduser('~')
    OUTPUT_DIR = f"{HOME}/vllm/experiments/content_output_AIME"
    KVCACHE_USAGES_DIR = f"{HOME}/vllm/experiments/kvcache_usages_AIME"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(KVCACHE_USAGES_DIR, exist_ok=True)
    
    return {
        'HOME': HOME,
        'OUTPUT_DIR': OUTPUT_DIR,
        'KVCACHE_USAGES_DIR': KVCACHE_USAGES_DIR,
        'USAGE_FILE': f"{HOME}/vllm/experiments/kvcache_usages/kvcache_usage.csv",
        'CSV_FILE': f"{HOME}/vllm/experiments/token_counts_AIME.csv"
    }

def load_data():
    """Load and prepare the AIME_2024 dataset."""
    df = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    # df = pd.read_parquet("hf://datasets/Maxwell-Jia/AIME_2024/aime_2024_problems.parquet")  #Old, incorrect way
    examples = [row for row in df]
    return examples


def setup_tokenizer(home_dir):
    """Initialize and return the tokenizer."""
    chat_tokenizer_dir = f"{home_dir}/vllm/experiments/deepseek_tokenizer"  # Or any other suitable tokenizer
    return transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
    )

def extract_reasoning_text(content):
    """
    Extract the reasoning section from the content.
    
    Since the reasoning text no longer starts with a <think> tag,
    start extraction from the beginning until the end tag </think>.
    If the end tag is missing, return the entire content.
    """
    end_tag = "</think>"
    end_index = content.find(end_tag)
    if end_index == -1:
        return content
    return content[:end_index]

def count_reasoning_tokens(content, tokenizer):
    """Count the number of tokens in the reasoning section."""
    reasoning_text = extract_reasoning_text(content)
    if reasoning_text is None:
        return 0
    return len(tokenizer.encode(reasoning_text))

def count_thoughts_positions(content, tokenizer):
    """
    Count occurrences of target phrases in the reasoning section 
    and return the count and positions.
    """
    reasoning_text = extract_reasoning_text(content)
    if reasoning_text is None:
        return 0, []
    
    # Tokenize reasoning text with offsets mapping
    tokenized = tokenizer(reasoning_text, 
                        add_special_tokens=False, 
                        return_offsets_mapping=True)
    offsets = tokenized['offset_mapping']

    thought_positions = []
    target_phrases = ["alternative", "Alternative", "Another", "But another", "Wait", "Oh wait"]
    for phrase in target_phrases:
        search_start = 0
        while True:
            pos = reasoning_text.find(phrase, search_start)
            if pos == -1:
                break
            # Find the token that covers the starting position (first token of phrase)
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start <= pos < token_end:
                    thought_positions.append(i)
                    break
            search_start = pos + 1

    thought_positions.insert(0, 0)  # Add the position of the original thought
    thought_positions.sort()
    thought_count = len(thought_positions)
    return thought_count, thought_positions

def count_non_reasoning_tokens(content, tokenizer):
    """Count tokens after the reasoning section."""
    end_tag = "</think>"
    end_index = content.find(end_tag) + len(end_tag)
    if end_index != -1:
        non_reasoning_text = content[end_index:]
        tokenized_result = tokenizer.encode(non_reasoning_text)
        return len(tokenized_result)
    return 0

def create_prompt(example):
    """Create a prompt from the example."""
    prompt = f"""
    Answer the following question.
    {example['Problem']}
    """.strip()

    return prompt, example["Answer"]

def process_example(example, iteration, paths, tokenizer, model, ip_address):
    """Process a single example and return the results."""
    prompt, correct_answer = create_prompt(example)
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # Construct URL using the provided IP address
    url = f"http://{ip_address}:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=data).json()
    content = response["choices"][0]["message"]["content"]

    # Write question and content to file
    with open(f"{paths['OUTPUT_DIR']}/question_{iteration}.txt", "w") as text_file:
        text_file.write(f"Question: {example['Problem']}\n\n")
        text_file.write("Assistant Response:\n")
        text_file.write(content)
        text_file.write(f"\n\nCorrect Answer: {correct_answer}")

    # Extract token usage information
    usage = response['usage']
    prompt_tokens = usage['prompt_tokens']
    total_tokens = usage['total_tokens']
    completion_tokens = usage['completion_tokens']

    # Determine score (This part might need adjustments based on how you want to evaluate)
    #Simple string matching, with some flexibility
    score = 1 if str(correct_answer).strip().lower() in content.strip().lower() else 0

    # Count tokens
    reasoning_tokens = count_reasoning_tokens(content, tokenizer)
    non_reasoning_tokens = count_non_reasoning_tokens(content, tokenizer)
    thought_count, thought_positions = count_thoughts_positions(content, tokenizer)

    # Handle KV cache usage file
    handle_kvcache_file(iteration, paths)
    
    return {
        'prompt_tokens': prompt_tokens,
        'total_tokens': total_tokens,
        'completion_tokens': completion_tokens,
        'reasoning_tokens': reasoning_tokens,
        'non_reasoning_tokens': non_reasoning_tokens, 
        'thought_count': thought_count,
        'thought_positions': thought_positions,
        'score': score
    }

def handle_kvcache_file(iteration, paths):
    """Handle the KV cache usage file."""
    usage_file = paths['USAGE_FILE']
    new_usage_file = f"{paths['KVCACHE_USAGES_DIR']}/question_{iteration}.csv"
    if os.path.exists(usage_file):
        shutil.copyfile(usage_file, new_usage_file)
        os.remove(usage_file)

def run_evaluation(start_iteration=1, end_iteration=None, iterations=None, ip_address="localhost", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"):
    """Run the evaluation process.

    If 'iterations' (a list of iteration numbers) is provided, only those examples are processed.
    Otherwise, examples between start_iteration and end_iteration (inclusive) are processed.
    """
    paths = setup_directories()
    examples = load_data()
    print(f"Loaded {len(examples)} examples.")
    tokenizer = setup_tokenizer(paths['HOME'])
    
    results = []
    
    if iterations is not None:
        # Process only specific iterations provided by the user
        valid_iterations = []
        for iteration in iterations:
            if 1 <= iteration <= len(examples):
                valid_iterations.append(iteration)
            else:
                print(f"Warning: iteration {iteration} is out of bounds, skipping.")
        for iteration in tqdm(valid_iterations, desc="Processing specific questions"):
            example = examples[iteration - 1]
            result = process_example(example, iteration, paths, tokenizer, model, ip_address)
            results.append((iteration, result))
    else:
        # Set end_iteration to process all examples if not specified
        if end_iteration is None or end_iteration > len(examples):
            end_iteration = len(examples)
            
        # Validate iteration range
        if start_iteration < 1:
            start_iteration = 1
        if start_iteration > end_iteration:
            print(f"Error: start_iteration ({start_iteration}) > end_iteration ({end_iteration})")
            return
        
        # Adjust for 0-based indexing: enumerate iterations from start_iteration to end_iteration
        for i in tqdm(range(start_iteration, end_iteration + 1), desc="Processing questions"):
            example = examples[i - 1]
            result = process_example(example, i, paths, tokenizer, model, ip_address)
            results.append((i, result))
    
    # Write results to CSV
    with open(paths['CSV_FILE'], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Iteration', 'Prompt Tokens', 'Total Tokens', 'Completion Tokens',
            'Reasoning Tokens', 'Non-Reasoning Tokens', 'Thought Count',
            'Thought Positions', 'Score'
        ])
        for iteration, res in results:
            writer.writerow([
                iteration,
                res['prompt_tokens'], 
                res['total_tokens'], 
                res['completion_tokens'],
                res['reasoning_tokens'], 
                res['non_reasoning_tokens'], 
                res['thought_count'],
                res['thought_positions'], 
                res['score']
            ])

def main():
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate LLM performance on AIME_2024 dataset.')
    parser.add_argument('--start', type=int, default=1, 
                        help='Starting iteration (1-indexed, default: 1)')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending iteration (inclusive, default: process all examples)')
    parser.add_argument('--iterations', type=str, default=None,
                        help='Comma-separated list of specific iterations to run (overrides --start and --end)')
    parser.add_argument('--ip', type=str, default="localhost",
                        help='IP address of the server for chat completions endpoint (default: localhost)')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help='Model identifier for the chat completions (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)')
    
    args = parser.parse_args()
    
    if args.iterations:
        try:
            iterations = [int(x.strip()) for x in args.iterations.split(',') if x.strip()]
        except ValueError:
            print("Error: Unable to parse iterations. Provide comma-separated integers.")
            return
        run_evaluation(iterations=iterations, ip_address=args.ip, model=args.model)
    else:
        run_evaluation(args.start, args.end, ip_address=args.ip, model=args.model)

if __name__ == "__main__":
    main()