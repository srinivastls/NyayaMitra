import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Load dataset
csv_path = "cleaned_combined_output.csv"  # Update with your file path
df = pd.read_csv(csv_path)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "microsoft/Phi-3-mini-128k-instruct"  # Update if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["judgment"], truncation=True, padding="max_length", max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# LoRA Config
lora_config = LoraConfig(
    r=8,  # Low-rank adaptation dimension
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj"],  # Fine-tune attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training Arguments
training_args = TrainingArguments(
    output_dir="./phi3_legal_finetune1",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False
)


# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    args=training_args,
)


# Start Training
trainer.train()

# Save Model
trainer.save_model("./phi3_legal_finetuned1")
tokenizer.save_pretrained("./phi3_legal_finetuned1")
