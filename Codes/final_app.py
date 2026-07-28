from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()
from peft import PeftModel
# Load the phi-3-mini-128k-instruct model
BASE_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
ADAPTER_PATH = "srinivastl/Nyaya" 

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True,torch_dtype=torch.float32)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)


#model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    chat_input = data.get("prompt", "")
    prompt = f"""[INSTRUCTION]
You are an AI legal assistant trained on Indian law. Provide clear, concise, and legally accurate responses based on Indian legal statutes and case law.
- Use simple language.
- Cite relevant legal sections when possible.
- Avoid personal opinions or unofficial advice.
- Maintain a professional tone.
- Answer within 100-200 words.

[USER QUERY]
{chat_input}

[RESPONSE]
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}