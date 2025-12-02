#!/usr/bin/env python3
"""
Download and save CommonsenseQA dataset to local directory
"""
from datasets import load_dataset
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Set output directory
output_dir = 'data/commonsense_qa'
os.makedirs(output_dir, exist_ok=True)

print("Loading CommonsenseQA dataset from cache or downloading...")
try:
    # Try to load from cache first
    dataset = load_dataset('tau/commonsense_qa')
    print(f"Dataset loaded successfully!")
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Save to local directory
    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved successfully to {output_dir}!")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nIf network error occurs, the dataset might already be in cache.")
    print("Cache location: ~/.cache/huggingface/datasets/tau___commonsense_qa/")
