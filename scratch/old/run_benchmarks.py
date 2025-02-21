import json
import time
import csv
import requests
import os
import shutil

def run_benchmarks(
    prompts_file="prompts.json",
    endpoint="http://localhost:8000/v1/chat/completions"
):
    # Load prompts
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    iteration = 1
    for prompt_data in prompts:
        payload = {
            "model": prompt_data.get("model","deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "messages":[{"role":"user","content":prompt_data["content"]}]
        }
        response = requests.post(endpoint, json=payload, timeout=120)
        print(response.json()['choices'][0]['message']['content'][-50:])
        print(response.json()['usage'])
        usage_file = "/home/colin/vllm/scratch/kvcache_usage.csv"
        new_usage_file = f"/home/colin/vllm/scratch/question_{iteration}.csv"
        if os.path.exists(usage_file):
            shutil.copyfile(usage_file, new_usage_file)
            os.remove(usage_file)
        iteration += 1

if __name__=="__main__":
    run_benchmarks()
