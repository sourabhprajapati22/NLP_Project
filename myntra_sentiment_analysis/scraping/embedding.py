from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from pathlib import Path
import os
import json
from tqdm import tqdm

# File path setup
file_path = Path(__file__).resolve().parent.parent
file_name = 'reviews_same.csv'
comp_file_name = file_path / f'data/processed/{file_name}'

# Read data
df = pd.read_csv(comp_file_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# Output path
output_file = file_path / 'data/processed/review_embeddings_flat.jsonl'

with open(output_file, 'w') as f:
    for idx, row in tqdm(df.iterrows()):
        product_id = row['product_id']
        sentence = row['cleaned_review_without_emoji']
        sentiment = row['sentiment_label']  # <-- Add this line

        # Tokenize and move to device
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
        
        # Create flattened dict
        record = {'product_id': product_id}
        for i, val in enumerate(embedding):
            record[f'emb_{i}'] = val

        record['sentiment_label'] = sentiment  # <-- Add sentiment at the end

        # Save as json line
        f.write(json.dumps(record) + '\n')

print(f" Saved flattened embeddings to: {output_file}")
