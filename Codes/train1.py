from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import SFTTrainer
from datasets import load_dataset
import numpy as np
import torch
import evaluate


# Load tokenizer and model
model_name = "microsoft/Phi-3-mini-128k-instruct"  # Update if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Load new dataset from CSV
new_data_path = "cleaned_combined_output.csv"  # Update with your actual CSV file path
new_dataset = load_dataset("csv", data_files=new_data_path)

# Ensure the column name is correct (adjust based on your CSV structure)
TEXT_COLUMN = "judgment"  # Update this if your text is stored under a different column name

# ✅ Split dataset into train (90%) and validation (10%)
dataset_split = new_dataset["train"].train_test_split(test_size=0.1)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples[TEXT_COLUMN], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset_split.map(tokenize_function, batched=True)

# ✅ Load evaluation metrics (Perplexity & Accuracy)
perplexity_metric = evaluate.load("perplexity")
accuracy_metric = evaluate.load("accuracy")

# ✅ Function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Convert logits to loss (Perplexity calculation)
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding tokens
    loss = loss_fct(logits_tensor.view(-1, logits_tensor.size(-1)), labels_tensor.view(-1)).item()
    perplexity = np.exp(loss)

    # Accuracy: Compare predicted tokens vs actual tokens (ignoring padding)
    non_pad_mask = labels_tensor != tokenizer.pad_token_id
    correct = (predictions == labels_tensor) & non_pad_mask
    accuracy = correct.sum().item() / non_pad_mask.sum().item()

    return {"perplexity": perplexity, "accuracy": accuracy}

# ✅ Training arguments with evaluation every 10,000 or 20,000 steps
training_args = TrainingArguments(
    output_dir="./phi3_legal_finetune3",  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",  # ✅ Evaluate every X steps
    eval_steps=10000,  # Change to 20000 if needed
    save_strategy="steps",
    save_steps=50000,  # Save model every 10,000 steps
    logging_dir="./logs",
    logging_steps=500,  # Log progress every 500 steps
    save_total_limit=2,
    num_train_epochs=3,  
    push_to_hub=False,
    eval_accumulation_steps=10
)

# ✅ Trainer with evaluation enabled
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Include validation set for evaluation
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,  # Add evaluation function
)

# ✅ Resume Training with Evaluation
trainer.train()

# ✅ Save Modelz
trainer.save_model("./phi3_legal_finetuned2")
tokenizer.save_pretrained("./phi3_legal_finetuned2")
