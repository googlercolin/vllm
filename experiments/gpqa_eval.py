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

def setup_directories():
    """Set up necessary directories and return path constants."""
    HOME = os.path.expanduser('~')
    OUTPUT_DIR = f"{HOME}/vllm/experiments/content_output"
    KVCACHE_USAGES_DIR = f"{HOME}/vllm/experiments/kvcache_usages"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(KVCACHE_USAGES_DIR, exist_ok=True)
    
    return {
        'HOME': HOME,
        'OUTPUT_DIR': OUTPUT_DIR,
        'KVCACHE_USAGES_DIR': KVCACHE_USAGES_DIR,
        'USAGE_FILE': f"{KVCACHE_USAGES_DIR}/kvcache_usage.csv",
        'CSV_FILE': f"{HOME}/vllm/experiments/token_counts.csv"
    }

def load_data():
    """Load and prepare the dataset."""
    # Login using e.g. `huggingface-cli login` to access this dataset
    df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")
    rng = random.Random(0)
    
    examples = [row.to_dict() for _, row in df.iterrows()]
    # Attach a random permutation (0..3) to each example
    examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
    
    return examples

def setup_tokenizer(home_dir):
    """Initialize and return the tokenizer."""
    chat_tokenizer_dir = f"{home_dir}/vllm/experiments/tokenizer"
    return transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
    )

def extract_reasoning_text(content):
    """
    Extract the reasoning section from the content.
    Returns the reasoning text or None if not found.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = content.find(start_tag)
    if start_index == -1:
        return None
    start_index += len(start_tag)
    end_index = content.find(end_tag, start_index)
    if end_index == -1:
        return None
    return content[start_index:end_index]

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
    target_phrases = ["alternative", "Alternative", "Another", "But another"]
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
    # Randomly permute the four options
    choices = [
        example["Correct Answer"],
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"],
    ]
    choices = [choices[idx] for idx in example["permutation"]]

    # Identify which choice is correct
    correct_index = choices.index(example["Correct Answer"])
    correct_answer = "ABCD"[correct_index]

    # Build the multiple-choice prompt
    prompt = f"""
    Answer the following multiple choice question. 
    The last line of your response should be of the format: 
    'Answer: $LETTER'{example['Question']}
    
    A) {choices[0]}
    B) {choices[1]}
    C) {choices[2]}
    D) {choices[3]}
    """.strip()

    return prompt, choices, correct_answer

def process_example(example, iteration, paths, tokenizer):
    """Process a single example and return the results."""
    prompt, choices, correct_answer = create_prompt(example)
    
    data = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # URL and headers for the POST request
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=data).json()
    content = response["choices"][0]["message"]["content"]

    # Write question, choices, and content to file
    with open(f"{paths['OUTPUT_DIR']}/question_{iteration}.txt", "w") as text_file:
        text_file.write(f"Question: {example['Question']}\n")
        text_file.write(f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\n")
        text_file.write("Assistant Response:\n")
        text_file.write(content)

    # Extract token usage information
    usage = response['usage']
    prompt_tokens = usage['prompt_tokens']
    total_tokens = usage['total_tokens']
    completion_tokens = usage['completion_tokens']

    # Determine score
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, content)
    extracted_answer = match.group(1) if match else None
    score = 1 if extracted_answer == correct_answer else 0

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

def run_evaluation(start_iteration=1, end_iteration=None):
    """Run the evaluation process."""
    paths = setup_directories()
    examples = load_data()
    tokenizer = setup_tokenizer(paths['HOME'])
    
    # Set end_iteration to process all examples if not specified
    if end_iteration is None or end_iteration > len(examples):
        end_iteration = len(examples)
        
    # Validate iteration range
    if start_iteration < 1:
        start_iteration = 1
    if start_iteration > end_iteration:
        print(f"Error: start_iteration ({start_iteration}) > end_iteration ({end_iteration})")
        return
    
    # Adjust for 0-based indexing
    examples_to_process = examples[start_iteration-1:end_iteration]
    
    # Open CSV file to write results
    with open(paths['CSV_FILE'], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Prompt Tokens', 'Total Tokens', 'Completion Tokens',
            'Reasoning Tokens', 'Non-Reasoning Tokens', 'Thought Count',
            'Thought Positions', 'Score'
        ])

        # Process each example
        for i, example in enumerate(tqdm(examples_to_process, desc="Processing questions")):
            iteration = start_iteration + i
            result = process_example(example, iteration, paths, tokenizer)
            
            writer.writerow([
                result['prompt_tokens'], 
                result['total_tokens'], 
                result['completion_tokens'],
                result['reasoning_tokens'], 
                result['non_reasoning_tokens'], 
                result['thought_count'],
                result['thought_positions'], 
                result['score']
            ])

def main():
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate LLM performance on GPQA dataset.')
    parser.add_argument('--start', type=int, default=1, 
                        help='Starting iteration (1-indexed, default: 1)')
    parser.add_argument('--end', type=int, default=None,
                        help='Ending iteration (inclusive, default: process all examples)')
    
    args = parser.parse_args()
    run_evaluation(args.start, args.end)

if __name__ == "__main__":
    main()