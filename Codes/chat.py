from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the tokenizer and base model
base_model_name = "Srinivastl/NyayaMitra"
model = AutoModelForCausalLM.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)



# Load the PEFT model (assuming you have the PEFT model saved at 'adapter_path')
# In this case, we'll assume no custom adapter path, but you can specify one if needed.
adapter_path = "Srinivastl/NyayaMitra_inst"  # specify adapter path if you have one
peft_model = PeftModel.from_pretrained(model, adapter_path)

# Set the model to evaluation mode
peft_model.eval()

# Function to chat with the model
def chat_with_model():
    print("Chat with the model (type 'exit' to stop)...")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting chat.")
            break
        
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # Generate model response
        with torch.no_grad():
            outputs = peft_model.generate(inputs["input_ids"], max_length=150, pad_token_id=tokenizer.eos_token_id)
        
        # Decode and display the model's response
        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model: {model_output}")

# Run the chat
chat_with_model()
