from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("nisaar/Lawyer_GPT_India", split="train")

# Load model & tokenizer
model_name = "srinivastl/NyayaMitra"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format dataset
dataset = dataset.map(lambda x: {"text": f"User: {x['question']}\nBot: {x['answer']}"})

# Tokenize dataset
dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512), batched=True)

# Set training args
training_args = TrainingArguments(
    output_dir="./nyaya-legal-chatbot",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    fp16=True  # Disable if no GPU
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Fine-tune
trainer.train()

# Save model
trainer.save_model("./nyaya-legal-chatbot")
tokenizer.save_pretrained("./nyaya-legal-chatbot")
