from transformers import AutoModelForCausalLM, AutoTokenizer
save_dir = "phi3-lora-finetuned"  # Directory where the model is saved
from peft import PeftModel



# Load saved model
model = AutoModelForCausalLM.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
from huggingface_hub import login

# Replace 'your_token_here' with your Hugging Face API token
login('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')

# Push to Hugging Face Hub
model.push_to_hub("Srinivastl/Nyayamitra_inst")
tokenizer.push_to_hub("srinivastl/Nyayamitra_inst")

print("Model successfully uploaded to Hugging Face!")
