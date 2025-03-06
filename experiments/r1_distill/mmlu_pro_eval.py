import pandas as pd
import os
import requests
import transformers
from tqdm import tqdm
import random
import re
import argparse
from datasets import load_dataset

def setup_directories():
    """Set up necessary directories."""
    home_dir = os.path.expanduser('~')
    base_dir = os.path.join(home_dir, 'vllm', 'experiments')
    output_dir = os.path.join(base_dir, 'content_output')
    kvcache_usages_dir = os.path.join(base_dir, 'kvcache_usages')
    usage_file = os.path.join(base_dir, 'kvcache_usage.csv')
    csv_file = os.path.join(base_dir, 'token_counts.csv')
    chat_tokenizer_dir = os.path.join(base_dir, 'r1_distill', 'tokenizer')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(kvcache_usages_dir, exist_ok=True)
    
    return home_dir, output_dir, kvcache_usages_dir, usage_file, csv_file, chat_tokenizer_dir

def load_data(category=None):
    """Load and prepare the MMLU-Pro dataset, optionally filtering by category."""
    df = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(0)

    if category:
        df = df.filter(lambda example: example["category"] == category)

    examples = [row for row in df]
    examples = [example | {"permutation": rng.sample(range(10), 10)} for example in examples]

    return examples

def setup_tokenizer(chat_tokenizer_dir):
    """Initialize and return the tokenizer."""
    return transformers.AutoTokenizer.from_pretrained(
        chat_tokenizer_dir, trust_remote_code=True
    )

def create_prompt(example):
    """Create a prompt from the MMLU-Pro example."""
    choices = example['options']
    choices = [choices[i] for i in example["permutation"]]
    correct_index = example["permutation"].index(example['answer'])
    correct_answer = "ABCDEFGHIJ"[correct_index]

    prompt = f"""
    Answer the following multiple choice question. 
    The last line of your response should be of the format: 
    'Answer: $LETTER'{example['question']}

    A) {choices[0]}
    B) {choices[1]}
    C) {choices[2]}
    D) {choices[3]}
    E) {choices[4]}
    F) {choices[5]}
    G) {choices[6]}
    H) {choices[7]}
    I) {choices[8]}
    J) {choices[9]}
    """.strip()

    return prompt, choices, correct_answer

def make_api_request(prompt, ip_address, model):
    """Makes the API request and returns the response."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    url = f"http://{ip_address}:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None

def extract_content_and_usage(response):
    """Extracts content and usage data from the API response."""
    if not response or "choices" not in response or not response["choices"]:
        return None, None, None, None

    content = response["choices"][0]["message"]["content"]
    usage = response['usage']
    prompt_tokens = usage['prompt_tokens']
    total_tokens = usage['total_tokens']
    completion_tokens = usage['completion_tokens']
    return content, prompt_tokens, total_tokens, completion_tokens

def extract_reasoning_text(content):
    """Extracts the reasoning part from the LLM output."""
    end_tag = "</think>"
    end_index = content.find(end_tag)
    return content[:end_index] if end_index != -1 else content

def count_reasoning_tokens(reasoning_text, tokenizer):
    """Counts reasoning tokens."""
    return len(tokenizer.encode(reasoning_text)) if reasoning_text else 0

def count_thoughts_positions(reasoning_text, tokenizer):
    """Counts and finds positions of thought phrases."""
    if not reasoning_text:
        return 0, []

    tokenized = tokenizer(reasoning_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized['offset_mapping']
    thought_positions = []
    target_phrases = ["alternative", "Alternative", "Another", "But another", "Wait", "Oh wait"]

    for phrase in target_phrases:
        search_start = 0
        while True:
            pos = reasoning_text.find(phrase, search_start)
            if pos == -1:
                break
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start <= pos < token_end:
                    thought_positions.append(i)
                    break
            search_start = pos + 1

    thought_positions.insert(0, 0)
    thought_positions.sort()
    return len(thought_positions), thought_positions

def count_non_reasoning_tokens(content, tokenizer):
    """Counts non-reasoning tokens."""
    end_tag = "</think>"
    end_index = content.find(end_tag)
    if end_index != -1:
        non_reasoning_text = content[end_index + len(end_tag):]
        return len(tokenizer.encode(non_reasoning_text))
    return 0

def calculate_score(content, correct_answer):
    """Calculates the score (0 or 1) based on the extracted answer."""
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-J])\$?"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, content)
    extracted_answer = match.group(1) if match else None
    return 1 if extracted_answer == correct_answer else 0
    
def handle_kvcache_file(iteration, kvcache_usages_dir, usage_file):
    """Copies and removes the KV cache usage file."""
    new_usage_file = os.path.join(kvcache_usages_dir, f"question_{iteration}.csv")
    try:
      if os.path.exists(usage_file):
          with open(usage_file, 'r') as src, open(new_usage_file, 'w') as dst:
              dst.write(src.read())  # More efficient file copying
          os.remove(usage_file)
    except OSError as e:
        print(f"Error handling KV cache file: {e}")

def process_example(example, iteration, output_dir, kvcache_usages_dir, usage_file, tokenizer, model, ip_address):
    """Process a single example."""
    prompt, choices, correct_answer = create_prompt(example)
    response = make_api_request(prompt, ip_address, model)

    if response is None:
        return None  # Handle API request failure

    content, prompt_tokens, total_tokens, completion_tokens = extract_content_and_usage(response)
    if content is None:
        return None  # Handle missing content

    # Write question, choices, and content to file
    try:
        with open(os.path.join(output_dir, f"question_{iteration}.txt"), "w") as text_file:
            text_file.write(f"Question: {example['question']}\n")
            for i, choice in enumerate(choices):
                text_file.write(f"{chr(65 + i)}) {choice}\n")
            text_file.write("\nAssistant Response:\n")
            text_file.write(content)
    except OSError as e:
        print(f"Error writing output file: {e}")
        return None # Return None if writing fails

    reasoning_text = extract_reasoning_text(content)
    reasoning_tokens = count_reasoning_tokens(reasoning_text, tokenizer)
    thought_count, thought_positions = count_thoughts_positions(reasoning_text, tokenizer)
    non_reasoning_tokens = count_non_reasoning_tokens(content, tokenizer)
    score = calculate_score(content, correct_answer)
    handle_kvcache_file(iteration, kvcache_usages_dir, usage_file)
    
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


def run_evaluation(start_iteration=1, end_iteration=None, iterations=None, ip_address="localhost", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", category="philosophy"):
    """Runs the evaluation."""
    home_dir, output_dir, kvcache_usages_dir, usage_file, csv_file, chat_tokenizer_dir = setup_directories()
    examples = load_data(category=category)
    tokenizer = setup_tokenizer(chat_tokenizer_dir)

    results = []
    
    if iterations:
      iterations_to_process = [iter_num for iter_num in iterations if 1 <= iter_num <= len(examples)]
    else:
      start_iteration = max(1, start_iteration)
      end_iteration = min(len(examples), end_iteration) if end_iteration else len(examples)
      if start_iteration > end_iteration:
          print("Error: start_iteration cannot be greater than end_iteration")
          return
      iterations_to_process = range(start_iteration, end_iteration + 1)

    for i in tqdm(iterations_to_process, desc="Processing questions"):
    example = examples[i-1]
    result = process_example(example, i, output_dir, kvcache_usages_dir, usage_file, tokenizer, model, ip_address)
    if result:
        results.append((i, result))
    
    # Write results using pandas
    df = pd.DataFrame([res[1] for res in results], index=[res[0] for res in results])
    df.index.name = 'Iteration'  # Set the index name
    df.to_csv(csv_file) # Write to the csv file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Evaluate LLM on MMLU-Pro.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed)')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive)')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated specific iterations')
    parser.add_argument('--ip', type=str, default="localhost", help='IP address of the server')
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help='Model identifier')
    parser.add_argument('--category', type=str, default="philosophy", help='MMLU-Pro category (case-sensitive)')

    args = parser.parse_args()

    if args.iterations:
        try:
            iterations = [int(x.strip()) for x in args.iterations.split(',') if x.strip()]
            run_evaluation(iterations=iterations, ip_address=args.ip, model=args.model, category=args.category)
        except ValueError:
            print("Error: Invalid iterations. Provide comma-separated integers.")
    else:
        run_evaluation(start_iteration=args.start, end_iteration=args.end, ip_address=args.ip, model=args.model, category=args.category)

if __name__ == "__main__":
    main()