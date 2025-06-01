import pandas as pd
from transformers import AutoTokenizer
import torch

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CSV file containing legal judgments
csv_file = 'cleaned_combined_output.csv'  # Replace with your actual CSV file path
df = pd.read_csv(csv_file)

# Print the first few rows to understand the structure of the data
print(df.head())

# Assuming the CSV contains a column 'judgment_text' with the full legal judgment text
# Let's preprocess the judgment text
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

def preprocess_text(text):
    # Clean the text by removing newlines, tabs, etc.
    text = text.replace("\n", " ").replace("\t", " ")
    return text

df['cleaned_judgment'] = df['judgment'].apply(preprocess_text)


def split_text_into_chunks(text, chunk_size=1024, overlap_size=128):
    # Split the text into overlapping windows
    tokens = tokenizer(text)["input_ids"]
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap_size):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

# Example: Split the long legal judgment texts into manageable chunks
chunked_texts = []
for judgment in df['cleaned_judgment']:
    chunks = split_text_into_chunks(judgment, chunk_size=4096, overlap_size=512)
    chunked_texts.extend(chunks)  # Collect all chunks into a list


from datasets import Dataset

# Prepare the dataset for pretraining
chunked_dataset = Dataset.from_dict({"input_ids": chunked_texts})

# Pretraining setup
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

# Load the Phi-3 Mini model
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Move model to GPU if available
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./legal_model",  # Directory to save the pretrained model
    per_device_train_batch_size=4,
    num_train_epochs=3,  # Number of epochs to train, adjust as necessary
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,  # Save model every 1000 steps
    # Enable GPU acceleration for training
    fp16=True if torch.cuda.is_available() else False  # Mixed precision if GPU is available
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=chunked_dataset,
    # Ensure the Trainer uses GPU
    data_collator=None,  # You can specify a data collator if needed
)

# Start pretraining
trainer.train()

# Save the pretrained model
model.save_pretrained("./legal_model")
tokenizer.save_pretrained("./legal_model")
