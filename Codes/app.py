from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

# Define paths for the base model and the adapter
BASE_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
ADAPTER_PATH = "srinivastl/Nyaya" 

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32)

# Load the PEFT adapter into the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Set model to evaluation mode
model.eval()

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    chat_input = data.get("prompt", "")

    # Prepare the legal assistant prompt
    prompt = f"""
    You are a highly knowledgeable and professional legal assistant specializing in Indian law. 
    You are tasked with providing clear, concise, and accurate legal information to individuals seeking advice 
    related to Indian laws, legal procedures, and rights. Please assist the user by answering their legal queries 
    in a manner that is easy to understand and in line with the Indian legal system. 

    When answering questions, always ensure to:
    1. Reference relevant sections of Indian law where applicable.
    2. Provide information based on current legal standards and practices.
    3. Maintain a neutral, unbiased tone and avoid offering personal legal opinions.
    4. If the query involves detailed legal advice, advise the user to consult with a qualified legal professional.

    Now, feel free to respond to any legal inquiry:+ {chat_input}'\n'
    [RESPONSE]
    """

    # Tokenize the prompt and move to appropriate device
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a response from the model
    outputs = model.generate(**inputs, max_new_tokens=256)

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded_output.split("[RESPONSE]")[-1].strip()

    
    # Return just the response, no additional data
    return response

@app.post("/summarize")
async def summarize(request: Request):
    data = await request.json()
    chat_input = data.get("prompt", "")
    retrieved_docs = data.get("retrieved_docs", "") 

    # Step 2: Prepare the prompt for summarization
    if retrieved_docs:  # If relevant documents are found
        prompt = f"""
        [INSTRUCTION]
        You are an AI legal assistant trained on Indian law. Your task is to summarize legal information based on the retrieved documents.
        - Use clear and simple language.
        - Provide a concise summary of the retrieved information, focusing on the key points.
        - Ensure the summary is within 100-200 words.
        - Include relevant legal references from the documents when applicable.

        [RELEVANT DOCUMENTS]
        {retrieved_docs}

        [USER QUERY]
        {chat_input}

        [SUMMARY]
        """
    else:  # If no relevant documents are retrieved, use the model's domain knowledge
        prompt = f"""
        [INSTRUCTION]
        You are an AI legal assistant trained on Indian law. Your task is to provide a legal answer based on your existing knowledge of Indian law.
        - Use clear and simple language.
        - Ensure the response is legally accurate.
        - Provide relevant legal references from Indian law when applicable.
        - If you cannot find a direct reference, use general knowledge of Indian law.

        [USER QUERY]
        {chat_input}

        [RESPONSE]
        """

    # Step 3: Tokenize the prompt and move to appropriate device (GPU/CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Step 4: Generate a response from the model based on the retrieved documents or domain knowledge
    outputs = model.generate(**inputs, max_new_tokens=256)

    # Step 5: Decode the output and extract only the generated text (response)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Return just the response (summary/answer), no additional data
    return response

@app.post("/fake-news-detection")
async def fake_news_api_call(query):
    # Simulate a call to a news API
    # In a real scenario, you would use an actual API call here
    return f"Fake news response for query: {query}"