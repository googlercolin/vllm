import pandas as pd
import csv
import os
import shutil
import requests
import transformers
from tqdm import tqdm

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_csv("hf://datasets/Idavidrein/gpqa/gpqa_diamond.csv")

# Home directory
HOME = os.path.expanduser('~')

# CSV file to store the token counts
csv_file = f"{HOME}/vllm/scratch/token_counts.csv"

# Extract the "Question" column
questions = df["Question"]

# URL and headers for the POST request
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Initialize the tokenizer
chat_tokenizer_dir = f"{HOME}/vllm/scratch/tokenizer"
tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)

def extract_reasoning_tokens(content):
    """
    Extract the reasoning section from the content and tokenize it.
    Returns the tokenized reasoning text or an empty list if not found.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = content.find(start_tag) + len(start_tag)
    end_index = content.find(end_tag)

    if start_index != -1 and end_index != -1:
        reasoning_text = content[start_index:end_index]
        return tokenizer.encode(reasoning_text)
    return []

def extract_reasoning_text(content):
    """
    Extract the reasoning section from the content.
    Returns the reasoning text or None if not found.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = content.find(start_tag) + len(start_tag)
    end_index = content.find(end_tag)

    if start_index != -1 and end_index != -1:
        return content[start_index:end_index]
    return None

def count_reasoning_tokens(content):
    """
    Count the number of tokens in the reasoning section.
    """
    reasoning_text = extract_reasoning_text(content)
    if reasoning_text is None:
        return 0
    return len(tokenizer.encode(reasoning_text))

def count_thoughts_positions(content):
    """
    Count occurrences of target phrases in the reasoning section (extracted from content)
    and return:
    - thought_count: number of occurrences.
    - thought_positions: token indices corresponding to the first token of each occurrence.
    """
    reasoning_text = extract_reasoning_text(content)
    # Tokenize reasoning text with offsets mapping
    tokenized = tokenizer(reasoning_text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenized['input_ids']
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

    thought_count = len(thought_positions) + 1 # Original thought is counted in
    thought_positions.insert(0, 0) # Add the position of the original thought
    thought_positions.sort()

    return thought_count, thought_positions


# Function to count non-reasoning tokens
def count_non_reasoning_tokens(content):
    end_tag = "</think>"
    end_index = content.find(end_tag) + len(end_tag)
    if end_index != -1:
        non_reasoning_text = content[end_index:]
        tokenized_result = tokenizer.encode(non_reasoning_text)
        return len(tokenized_result)
    return 0

# Open a CSV file to append the results
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prompt Tokens', 'Total Tokens', 'Completion Tokens', 'Reasoning Tokens', 'Non-Reasoning Tokens', 'Thought Count', 'Thought Positions'])

    # Loop through each question and send a POST request
    iteration = 1
    for question in tqdm(questions[:10], desc="Processing questions"):
        data = {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        # Extract the 'content' and usage details
        content = response_json['choices'][0]['message']['content']
        with open(f"{HOME}/vllm/scratch/content_output.txt", "w") as text_file:
            text_file.write(content)
        usage = response_json['usage']
        prompt_tokens = usage['prompt_tokens']
        total_tokens = usage['total_tokens']
        completion_tokens = usage['completion_tokens']

        # Count reasoning and non-reasoning tokens
        reasoning_tokens = count_reasoning_tokens(content)
        non_reasoning_tokens = count_non_reasoning_tokens(content)
        thought_count, thought_positions = count_thoughts_positions(content)
        
        # Append the extracted information to the CSV file
        writer.writerow([prompt_tokens, total_tokens, completion_tokens, reasoning_tokens, non_reasoning_tokens, thought_count, thought_positions])

        # Handle kvcache_usage.csv
        KVCACHE_USAGES_DIR = f"{HOME}/vllm/scratch/kvcache_usages"
        usage_file = f"{KVCACHE_USAGES_DIR}/kvcache_usage.csv"
        new_usage_file = f"{KVCACHE_USAGES_DIR}/question_{iteration}.csv"
        if os.path.exists(usage_file):
            shutil.copyfile(usage_file, new_usage_file)
            os.remove(usage_file)
        iteration += 1