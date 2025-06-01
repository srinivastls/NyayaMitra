import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import evaluate
import torch
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

# Load dataset
df = pd.read_csv("dataset_final.csv")
df=df[:1000]
df["prompt"] = df.apply(lambda row: f"Act as Indian legal expert to Summarize: {row['Judgment']}\nSummary:", axis=1)
df["target"] = df["HeadNote"]
dataset = Dataset.from_pandas(df[["prompt", "target"]]) # Limit to 1000 samples for testing
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load tokenizer
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use eos token as padding

# Load quantized model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Add LoRA adapters for efficient fine-tuning (optional)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["qkv_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

# Preprocessing function for long texts
def preprocess(example):
    # Encode both the prompt and target, ensuring we don't exceed max_length
    inputs = tokenizer(
        example["prompt"],
        padding="longest",  # Padding to max_length
        truncation=True,  # Truncating if it exceeds max_length
        max_length=30000,  # Set max length to 128K tokens for inputs
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"],
            padding="longest",  # Padding to max_length
            truncation=True,  # Truncating if it exceeds max_length
            max_length=6000,  # Set max length to 128K tokens for target
        )
    # Ensure that the model receives the labels for supervised training
    inputs["labels"] = labels["input_ids"]
    return inputs

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_eval = eval_dataset.map(preprocess, batched=True)

# Load ROUGE metric for evaluation
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Training Arguments (Supervised Fine-Tuning)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=20,
    gradient_accumulation_steps=8,  # Simulate larger batches
)

# Data collator for language modeling (for causal LM)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start fine-tuning (Supervised Fine-Tuning)
trainer.train()
