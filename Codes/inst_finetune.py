import json
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel, LoraConfig, get_peft_model

# -------------------------------
# Step 1: Load tokenizer and model
# -------------------------------

base_model_name = "Srinivastl/NyayaMitra"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
)

# -------------------------------
# Step 2: Apply PEFT (LoRA) to the model
# -------------------------------

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank for LoRA adaptation
    lora_alpha=16,  # Scaling factor for LoRA
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none",  # No bias added to LoRA layers
    task_type="CAUSAL_LM",  # Causal language model
    target_modules=["self_attn.qkv_proj", "self_attn.o_proj"],
)

# Apply PEFT (LoRA) to the model
model = get_peft_model(model, lora_config)

# -------------------------------
# Step 3: Check trainable parameters
# -------------------------------
model.print_trainable_parameters()  # Should work now that the model is wrapped with PEFT

# Make sure gradients flow
model.enable_input_require_grads()

# -------------------------------
# Step 4: Load the Dataset
# -------------------------------

class InstructionDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line)
                instruction = item['instruction']
                for instance in item['instances']:
                    input_text = instance['instruction_with_input']
                    output_text = instance['output']
                    full_prompt = f"Instruction: {input_text}\n\n### Response:\n{output_text}"
                    self.data.append(full_prompt)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=512,  # You can adjust the max_length based on your dataset
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

dataset = InstructionDataset(file_path="core_data.jsonl", tokenizer=tokenizer)

# -------------------------------
# Step 5: Set TrainingArguments
# -------------------------------

training_args = TrainingArguments(
    output_dir="./phi3-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100,
    num_train_epochs=3,
    optim="adamw_torch",
    bf16=True,  # or use fp16=True if bf16 is not supported
    save_total_limit=2,
)

# -------------------------------
# Step 6: Data Collator
# -------------------------------

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM doesn't use masked language modeling
)

# -------------------------------
# Step 7: Initialize Trainer
# -------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# Step 8: Start Fine-tuning
# -------------------------------

trainer.train()

# -------------------------------
# Step 9: Save the Fine-tuned Model
# -------------------------------

trainer.model.save_pretrained("./phi3-lora-finetuned")
tokenizer.save_pretrained("./phi3-lora-finetuned")
