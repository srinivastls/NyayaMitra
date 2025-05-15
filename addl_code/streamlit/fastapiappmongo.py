from fastapi import FastAPI,Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from langchain_community.llms import Ollama
import re
import json 
from milvus_query import *
from fastapi.middleware.cors import CORSMiddleware
from mongo_query import *
sections_dict= json.load(open('/home/ubuntu/ram/code/streamlit/section_dict.json','r'))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Specify the methods you want to allow, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Specify the headers you want to allow
)
# Initialize the LLM with the desired model

model_name = "phi3"
llm_phi3 = Ollama(model=model_name)

import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def string_map(string):
    match = re.search(r"content='([^']*)'", string)
    if match:
        return match.group(1)
    else:
        ''


async def serve_data(query: str):
    for resp in llm.stream(query):
        try:
            if string_map(f"{resp}"):
                yield string_map(f"{resp}")
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            print("caught cancelled error")
            raise GeneratorExit

async def serve_data_phi3(query: str):
    for resp in llm_phi3.stream(query):
        try:
            if resp:
                yield f"{resp}"
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            print("caught cancelled error")
            raise GeneratorExit

        # if 'choices' in resp:
        #     if 'delta' in resp['choices'][0]:
        #         if 'content' in resp['choices'][0]['delta']:
        #             yield f"{resp['choices'][0]['delta']['content']}"

        

@app.get('/get-summary/')
async def stream_summary(query: str):
    pattern = r'\b\d+(?:-[A-Za-z]+)?\b'
    matches = re.findall(pattern, query)
    sections_result = []
    for i in matches:
        for j in list(sections_dict.keys()):
            if i in j:
                sections_result.append(sections_dict[j])

    if len(sections_result) > 0:
        search_prayer_text = '\n'.join(sections_result)
    else:
        sections_result  = sections_search(query,20) # semantic search with
        search_prayer_text = '\n'.join(sections_result)
    prompt = f"""
                Objective: Create a concise summary of the provided legal document related to the Hyderabad Municipality Corporation.

                Document Details:
                - Document Context: {search_prayer_text}
                - Key Section: {query}

                Instructions: 
                - Summarize the main points and contextual information from the key section. total word count should be less than 200 words.
              """

    try:
        return StreamingResponse(serve_data(prompt), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")


@app.post('/get-pending-prayers/')
async def stream(request: Request):
    # Extract the JSON body from the request
    data = await request.json()
    query = data.get('query')  # Get the 'query' from the JSON body
    response = data.get('response')  # Get the 'response' from the JSON body, if available
    page_number = data.get('page_number', 1)  # Default to page 1 if not provided
    page_size = data.get('page_size', 10)  # Default to 10 items per page if not provided

    # Ensure the query is provided
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if not response:
        response = ''

    return pending_prayers_data(query, page_number, page_size)

@app.post('/get-disposed-prayers/')
async def stream(request: Request):
    # Extract the JSON body from the request
    data = await request.json()
    query = data.get('query')  # Get the 'query' from the JSON body
    response = data.get('response')  # Get the 'response' from the JSON body, if available
    page_number = data.get('page_number', 1)  # Default to page 1 if not provided
    page_size = data.get('page_size', 10)  # Default to 10 items per page if not provided

    # Ensure the query is provided
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if not response:
        response = ''

    return disposed_prayers_data(query, page_number, page_size)

@app.post('/get-orders/')
async def stream(request: Request):
    # Extract the JSON body from the request
    data = await request.json()
    query = data.get('query')  # Get the 'query' from the JSON body
    response = data.get('response')  # Get the 'response' from the JSON body, if available
    page_number = data.get('page_number', 1)  # Default to page 1 if not provided
    page_size = data.get('page_size', 10)  # Default to 10 items per page if not provided

    # Ensure the query is provided
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if not response:
        response = ''

    return orders_data(query, page_number, page_size)

@app.post('/get-judgments/')
async def stream(request: Request):
    # Extract the JSON body from the request
    data = await request.json()
    query = data.get('query')  # Get the 'query' from the JSON body
    response = data.get('response')  # Get the 'response' from the JSON body, if available
    page_number = data.get('page_number', 1)  # Default to page 1 if not provided
    page_size = data.get('page_size', 10)  # Default to 10 items per page if not provided

    # Ensure the query is provided
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if not response:
        response = ''

    return judgments_data(query, page_number, page_size)


@app.post('/get-judgments-summary/')
async def judgments_summary(request: Request):
    data = await request.json()
    judgment = data.get('judgment')
    if not judgment:
        raise HTTPException(status_code=400, detail="judgment parameter is required")
    prompt = f"""
                Summarize the following legal judgment. The summary should include:
                1. Key case details (e.g., parties involved, court, and relevant dates).
                2. Final judgment or ruling.
                3. Key legal principles and reasoning used by the court.

                Ensure the summary is concise, clear, and under 200 words:

                judgment:' {judgment}' """
    
    try:
        return StreamingResponse(serve_data_phi3(prompt), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")



import uvicorn

if __name__ == "__main__":
    uvicorn.run("fastapiappmongo:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")








