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
    prompt = f""""You are a highly knowledgeable and professional legal assistant specializing in Indian law. "
        "You are tasked with providing clear, concise, and accurate legal information to individuals seeking advice "
        "related to Indian laws, legal procedures, and rights. Please assist the user by answering their legal queries "
        "in a manner that is easy to understand and in line with the Indian legal system. "
        "When answering questions, always ensure to:\n"
        "1. Reference relevant sections of Indian law where applicable.\n"
        "2. Provide information based on current legal standards and practices.\n"
        "3. Maintain a neutral, unbiased tone and avoid offering personal legal opinions.\n"
        "4. If the query involves detailed legal advice, advise the user to consult with a qualified legal professional.\n"
        "Now, feel free to respond to any legal inquiry: " + {chat_input} + "\n"
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}


@app.post("/summarize")
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