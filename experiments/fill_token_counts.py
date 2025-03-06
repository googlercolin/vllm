import pandas as pd
import os
import transformers
from tqdm import tqdm

def setup_tokenizer():
    """Initialize and return the tokenizer."""
    HOME = os.path.expanduser('~')
    chat_tokenizer_dir = f"{HOME}/vllm/experiments/deepseek_tokenizer"
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

def count_non_reasoning_tokens(content, tokenizer):
    """
    Count tokens after the reasoning section.
    
    Finds the end tag </think> in the content and tokenizes everything 
    that comes after it.
    """
    end_tag = "</think>"
    end_index = content.find(end_tag)
    if end_index == -1:
        return 0
    end_index += len(end_tag)
    non_reasoning_text = content[end_index:]
    tokenized_result = tokenizer.encode(non_reasoning_text)
    return len(tokenized_result)

def count_thoughts_positions(content, tokenizer):
    """
    Count occurrences of target phrases in the reasoning section
    and return the count and token positions.
    
    Tokenizes the reasoning text (starting from the beginning) and 
    searches for target phrases.
    """
    reasoning_text = extract_reasoning_text(content)
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
            # Find the token index covering this position
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start <= pos < token_end:
                    thought_positions.append(i)
                    break
            search_start = pos + 1

    thought_positions.insert(0, 0)  # original thought at position 0
    thought_positions.sort()
    thought_count = len(thought_positions)
    return thought_count, thought_positions

def main():
    HOME = os.path.expanduser('~')
    CSV_FILE = f"{HOME}/vllm/experiments/token_counts_copy.csv"
    OUTPUT_DIR = f"{HOME}/vllm/experiments/content_output"
    
    # Read the existing CSV file.
    df = pd.read_csv(CSV_FILE)
    
    tokenizer = setup_tokenizer()
    
    # Process each row (iteration) to fill up the missing columns.
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        iteration = row['Iteration']
        file_path = f"{OUTPUT_DIR}/question_{int(iteration)}.txt"
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            full_content = f.read()
        
        # Extracting the assistant response.
        parts = full_content.split("Assistant Response:\n", 1)
        if len(parts) < 2:
            print(f"Assistant response not found in {file_path}")
            continue
        assistant_response = parts[1]
        
        # Calculate non-reasoning tokens using the updated logic.
        new_non_reasoning_tokens = count_non_reasoning_tokens(assistant_response, tokenizer)
        # Calculation of reasoning tokens is: Completion Tokens - Non-Reasoning Tokens.
        # (Assuming "Completion Tokens" is already recorded in the CSV.)
        completion_tokens = row['Completion Tokens']
        new_reasoning_tokens = completion_tokens - new_non_reasoning_tokens
        
        # Calculate thought positions using the updated extraction.
        thought_count, thought_positions = count_thoughts_positions(assistant_response, tokenizer)
        
        # Update DataFrame columns.
        df.at[idx, 'Reasoning Tokens'] = new_reasoning_tokens
        df.at[idx, 'Thought Count'] = thought_count
        df.at[idx, 'Thought Positions'] = str(thought_positions)
    
    # Save the updated CSV.
    df.to_csv(CSV_FILE, index=False)
    print(f"Updated CSV file saved: {CSV_FILE}")

if __name__ == "__main__":
    main()