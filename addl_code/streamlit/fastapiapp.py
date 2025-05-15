from fastapi import FastAPI,Request, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from langchain_community.llms import Ollama
import re
import json 
from milvus_query import *
from fastapi.middleware.cors import CORSMiddleware
sections_dict= json.load(open('section_dict.json','r'))


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Specify the methods you want to allow, e.g., ["GET", "POST"]
    allow_headers=["*"],  # Specify the headers you want to allow
)
@app.get("/")
async def root():
    return {"message": "Welcome to the API"}
# Initialize the LLM with the desired model
model_name = "phi3"
llm_phi3 = Ollama(model=model_name)

import os
os.environ['OPENAI_API_KEY'] = ""
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def string_map(string):
    match = re.search(r"content='([^']*)'", string)
    if match:
        return match.group(1)
    else:
        ''


async def serve_data(query: str):
    for resp in llm_phi3.stream(query):
        try:
            yield string_map(f"{resp}")
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            print("caught cancelled error")
            raise GeneratorExit

async def serve_data_phi3(query: str):
    for resp in llm_phi3.stream(query):
        try:
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
    #pattern = r'\b\d+(?:-[A-Za-z]+)?\b'
    #matches = re.findall(pattern, query)
    #sections_result = []
    #for i in matches:
    #    for j in list(sections_dict.keys()):
    #        if i in j:
    #            sections_result.append(sections_dict[j])

    #if len(sections_result) > 0:
     #   search_prayer_text = '\n'.join(sections_result)
    #else:
    #    sections_result  = sections_search(query,20) # semantic search with
    #    search_prayer_text = '\n'.join(sections_result)
    #prompt = f"""
    #            Objective: Create a concise summary of the provided legal document related to the Hyderabad Municipality Corporation.

    #            Document Details:
    #            - Document Context: {search_prayer_text}
    #            - Key Section: {query}

    #            Instructions: 
    #            - Summarize the main points and contextual information from the key section. total word count should be less than 200 words.
    #          """

    try:
        prompt="write a python code for max number"
        return StreamingResponse(serve_data_phi3(prompt), media_type='text/event-stream')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")


@app.post('/get-prayers-judgements/')
async def stream(request: Request):
    # Extract the JSON body from the request
    data = await request.json()
    query = data.get('query')  # Get the 'query' from the JSON body
    response = data.get('response')
    # Ensure the query is provided
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if not response:
        response = ''
    
    prayer_result = prayers_search(query,10)

    search_judgeemnt_text = f'''section: {query}'''
    # Sample prayer:{response}
    # '''
    result_judgement = judgement_search(search_judgeemnt_text,10)

    orders = order_search(query,10)

    return {
        "prayers":prayer_result,
        "judgements":result_judgement,
        "orders": orders
    }


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
    uvicorn.run("fastapiapp:app", host="0.0.0.0", port=8000, reload=True, log_level="debug")








