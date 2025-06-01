import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import numpy as np

# Load your CSV file
csv_path = "cleaned_combined_output.csv"
df = pd.read_csv(csv_path)
df = df[:2000]  # Use first 2000 rows

# Load model and tokenizer
model_name = "Srinivastl/NyayaMitra"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tokenize and compute perplexity
def compute_perplexity_for_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())  # Perplexity = exp(loss)

# Compute perplexity for each row
df["perplexity"] = df["judgment"].apply(compute_perplexity_for_text)

# Summary statistics
average_perplexity = df["perplexity"].mean()
median_perplexity = df["perplexity"].median()
percentile_99 = np.percentile(df["perplexity"], 99)
percentile_95 = np.percentile(df["perplexity"], 95)
max_perplexity = df["perplexity"].max()
min_perplexity = df["perplexity"].min()

# Print results
print("----- Perplexity Summary -----")
print(f"Average Perplexity: {average_perplexity:.2f}")
print(f"Median Perplexity: {median_perplexity:.2f}")
print(f"99th Percentile: {percentile_99:.2f}")
print(f"95th Percentile: {percentile_95:.2f}")
print(f"Max Perplexity: {max_perplexity:.2f}")
print(f"Min Perplexity: {min_perplexity:.2f}")
