import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import asyncio
import re

def extract_answer(text):
    # Define the start and end markers
    start_marker = r"\[RESPONSE\]"
    end_marker_1 = r"\[END RESPONSE\]"
    end_marker_2 = r"END"
    
    print("Text to extract from:")
    print(text)
    
    # Create a regex pattern to match the text between the start marker and end markers
    pattern = rf"(?<={start_marker})(.*?)(?=(?:{end_marker_1}|{end_marker_2})|$)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(0).strip()
    else:
        # If no end marker is found, get everything after the start marker to the last possible position
        start_pos = text.find("[RESPONSE]")
        if start_pos == -1:
            return "Answer not found"
        
        # Try to find the last occurrence of either "END RESPONSE" or "END"
        end_pos_1 = text.rfind("[END RESPONSE]")
        end_pos_2 = text.rfind("END")
        
        # If both end markers are not found, return the whole text from the start marker to the end of the string
        if end_pos_1 == -1 and end_pos_2 == -1:
            return text[start_pos:].strip()
        
        # Find the last occurrence of either marker
        last_end_pos = max(end_pos_1, end_pos_2)
        return text[start_pos:last_end_pos].strip()

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="⚖️ NyayaMitra", layout="wide")

BASE_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
ADAPTER_PATH = "./phi3_legal_finetuned1" 

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

st.title("⚖️ NyayaMitra - AI Legal Assistant")

tab1, tab2 = st.tabs(["🤖 Chatbot", "📚 RAG"])

with tab1:
    st.subheader("🔹 Ask your legal question")
    chat_input = st.text_area("Enter your legal question:", key="chat_input")

    col1, col2 = st.columns([3, 1])
    with col2:
        submit_chat = st.button("Get Legal Advice")
    
    if submit_chat and chat_input.strip():
        instruction_prompt = f"""[INSTRUCTION]
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
        inputs = tokenizer(instruction_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs,max_length=1000)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        out=extract_answer(response)
        st.text_area("AI Response:", out, height=250)
    elif submit_chat:
        st.warning("Please enter a legal question!")


with tab2:
    st.subheader("Retrieve Legal Information")
    rag_input = st.text_input("Enter topic to retrieve information:", key="rag_input")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        submit_rag = st.button("Search")
    
    if submit_rag and rag_input.strip():
        st.write("AI: Here is some relevant legal information about your topic.")
    elif submit_rag:
        st.warning("Please enter a topic!")