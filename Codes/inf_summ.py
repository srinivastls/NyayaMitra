import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "microsoft/Phi-3-mini-128k-instruct"  # Replace with your pre-trained model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the CSV dataset
df = pd.read_csv('cleaned_combined_output.csv')  # Replace with the path to your CSV file
df = df[:200]  # Use a subset of the data for testing

# Ensure the column with text is named 'judgment', otherwise, modify the code accordingly
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]['judgment'])  # Replace 'judgment' with your actual column name
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()  # For language models, targets are the same as input_ids
        return {"input_ids": input_ids, "labels": labels}

# Create the dataset and dataloader
dataset = TextDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Function to calculate mean token accuracy
def mean_token_accuracy(model, tokenizer, dataloader, device):
    model.eval()
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating MTA"):
            # Move inputs and targets to the same device as the model (either cuda or cpu)
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            # Get model predictions
            outputs = model(input_ids=inputs, labels=targets)
            logits = outputs.logits

            # Get the predicted token indices from logits
            predictions = torch.argmax(logits, dim=-1)

            # Count correct tokens (where prediction matches target)
            correct_tokens = (predictions == targets).sum().item()
            total_correct += correct_tokens
            total_tokens += targets.numel()

    mean_accuracy = total_correct / total_tokens
    return mean_accuracy

# Compute the mean token accuracy
mta = mean_token_accuracy(model, tokenizer, dataloader, device)
print(f"Mean Token Accuracy: {mta * 100:.2f}%")
