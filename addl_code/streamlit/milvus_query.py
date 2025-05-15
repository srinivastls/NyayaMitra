from pymilvus import AnnSearchRequest
from langchain_community.embeddings import OllamaEmbeddings
from pymilvus import RRFRanker
from pymilvus import connections, Collection
import ast
import re
import json
from collections import defaultdict
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_milvus import Milvus

URI = "http://localhost:19530"

connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)

embeddings = OllamaEmbeddings(model="phi3")

model_name = "phi3"
llm = Ollama(model=model_name)



def prayers_search(search_text,limit=10):
    query_vector = embeddings.embed_query(search_text)
    search_param_1 = {
    "data": [query_vector], # Query vector
    "anns_field": "prayerVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": limit # Number of search results to return in this AnnSearchRequest
}
    req1 = AnnSearchRequest(**search_param_1)
    # Store these two requests as a list in `reqs`
    reqs = [req1]
    rerank = RRFRanker()
    collection = Collection(name="prayers_collection")
    collection.load()
    res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=10 # Number of final search results to return
)

    # The given data string
    data_str = str(res[0])+''  # Extract the string from the list
    data_list = ast.literal_eval(data_str)  # Convert the string to an actual list
    # Extracting IDs using regular expressions
    ids = [int(re.search(r'id: (\d+)', entry).group(1)) for entry in data_list]
    # Query Milvus for the documents with the specified IDs
    if len(ids)>0:
        results = collection.query(
            expr=f"prayerId in {ids}",  # Assuming "prayerId" is the field storing the IDs
            output_fields=["prayerId", "prayerText"]  # Specify the fields you want to retrieve
        )
        return [i['prayerText'] for i in results]
    else:
        return []


def prayers_search_2(search_text,limit=10):
    query_vector = embeddings.embed_query(search_text)
    search_param_1 = {
    "data": [query_vector], # Query vector
    "anns_field": "prayerVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": limit # Number of search results to return in this AnnSearchRequest
}
    req1 = AnnSearchRequest(**search_param_1)
    # Store these two requests as a list in `reqs`
    reqs = [req1]
    rerank = RRFRanker()
    collection = Collection(name="prayers_collection_2")
    collection.load()
    res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=10 # Number of final search results to return
)

    # The given data string
    data_str = str(res[0])+''  # Extract the string from the list
    data_list = ast.literal_eval(data_str)  # Convert the string to an actual list
    # Extracting IDs using regular expressions
    ids = [int(re.search(r'id: (\d+)', entry).group(1)) for entry in data_list]
    # Query Milvus for the documents with the specified IDs
    if len(ids)>0:
        results = collection.query(
            expr=f"prayerId in {ids}",  # Assuming "prayerId" is the field storing the IDs
            output_fields=["prayerId", "prayerText","CASE_NO","PETITIONER_ADV","RESPONDENT_ADV","PETITIONER","RESPONDENT"]  # Specify the fields you want to retrieve
        )
        return [{"prayerText":i['prayerText'],
                 "CASE_NO":i['CASE_NO'],
                 "PETITIONER_ADV":i['PETITIONER_ADV'],
                 "RESPONDENT_ADV":i['RESPONDENT_ADV'],
                 "PETITIONER":i['PETITIONER'],
                 "RESPONDENT":i['RESPONDENT']}
                for i in results]
    else:
        return []


def sections_search(search_text,limit=10):
    query_vector = embeddings.embed_query(search_text)
    search_param_1 = {
    "data": [query_vector], # Query vector
    "anns_field": "sectionVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": limit # Number of search results to return in this AnnSearchRequest
}
    req1 = AnnSearchRequest(**search_param_1)
    # Store these two requests as a list in `reqs`
    reqs = [req1]
    rerank = RRFRanker()
    collection = Collection(name="section_collection")
    collection.load()
    res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=10 # Number of final search results to return
)

    # The given data string
    data_str = str(res[0])+''  # Extract the string from the list
    data_list = ast.literal_eval(data_str)  # Convert the string to an actual list
    # Extracting IDs using regular expressions
    ids = [int(re.search(r'id: (\d+)', entry).group(1)) for entry in data_list]
    # Query Milvus for the documents with the specified IDs
    if len(ids)>0:
        results = collection.query(
            expr=f"id in {ids}",  # Assuming "prayerId" is the field storing the IDs
            output_fields=["id", "sectionText"]  # Specify the fields you want to retrieve
        )
        return [i['sectionText'] for i in results]
    else:
        return []



def clean_judgements(results):
    combined_text = defaultdict(lambda: defaultdict(str))
    for entry in results:
        filename = entry['judgementFileName']
        chunk_id = entry['judgementChunkId']
        text = entry['judgementText']
        combined_text[filename][chunk_id] += text + " "  # Add a space for separation between texts

    # Create a list to store the final combined texts
    result = []

    # Combine the chunks in the correct order and format the result
    for filename, chunks in combined_text.items():
        combined_texts = ''.join(chunks[i] for i in sorted(chunks.keys()))  # Sort chunks by chunk_id
        result.append({'judgementFileName': filename, 'combinedJudgementText': combined_texts.strip()})
    return result

def judgement_search(search_text,limit=10):
    query_vector = embeddings.embed_query(search_text)
    search_param_1 = {
    "data": [query_vector], # Query vector
    "anns_field": "judgementVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": limit # Number of search results to return in this AnnSearchRequest
}
    req1 = AnnSearchRequest(**search_param_1)
    # Store these two requests as a list in `reqs`
    reqs = [req1]
    rerank = RRFRanker()
    collection = Collection(name="judgement_collection")
    collection.load()
    res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=10 # Number of final search results to return
)

    # The given data string
    data_str = str(res[0])+''  # Extract the string from the list
    data_list = ast.literal_eval(data_str)  # Convert the string to an actual list
    # Extracting IDs using regular expressions
    ids = [int(re.search(r'id: (\d+)', entry).group(1)) for entry in data_list]
    # Query Milvus for the documents with the specified IDs
    if len(ids)>0:
        res_files = collection.query(
            expr=f"id in {ids}",  # Assuming "prayerId" is the field storing the IDs
            output_fields=["id",'judgementFileName']  # Specify the fields you want to retrieve
        )
        file_names = [i['judgementFileName'] for i in res_files]
        results = collection.query(
        expr=f"judgementFileName in {file_names}",  # Assuming "prayerId" is the field storing the IDs
        output_fields=["id", "judgementText","judgementChunkId",'judgementFileName'])
           

        return clean_judgements(results)
    else:
        return []

def clean_order(results):
    combined_text = defaultdict(lambda: defaultdict(str))
    other_data = {}
    for entry in results:
        filename = entry['FileName']
        chunk_id = entry['orderChunkId']
        text = entry['orderText']
        combined_text[filename][chunk_id] += text + " "  # Add a space for separation between texts
        if filename not in other_data:
            other_data[filename] = {
                "Judge":entry['Judge'],
                "Casedetails":entry['Casedetails'],
                "Date":entry['Date']

            }


    # Create a list to store the final combined texts

    result = []

    # Combine the chunks in the correct order and format the result
    for filename, chunks in combined_text.items():
        combined_texts = ''.join(chunks[i] for i in sorted(chunks.keys()))  # Sort chunks by chunk_id
        result.append({'FileName': filename, 'orderText': combined_texts.strip()})
    final_result = []
    for i in result:
        if i['FileName'] in other_data:
            i.update(other_data[i['FileName']])
            final_result.append(i)
    return final_result

def order_search(search_text,limit=10):
    query_vector = embeddings.embed_query(search_text)
    search_param_1 = {
    "data": [query_vector], # Query vector
    "anns_field": "orderVector", # Vector field name
    "param": {
        "metric_type": "L2", # This parameter value must be identical to the one used in the collection schema
        "params": {"nprobe": 10}
    },
    "limit": limit # Number of search results to return in this AnnSearchRequest
}
    req1 = AnnSearchRequest(**search_param_1)
    # Store these two requests as a list in `reqs`
    reqs = [req1]
    rerank = RRFRanker()
    collection = Collection(name="order_collection")
    collection.load()
    res = collection.hybrid_search(
    reqs, # List of AnnSearchRequests created in step 1
    rerank, # Reranking strategy specified in step 2
    limit=10 # Number of final search results to return
)

    # The given data string
    data_str = str(res[0])+''  # Extract the string from the list
    data_list = ast.literal_eval(data_str)  # Convert the string to an actual list
    # Extracting IDs using regular expressions
    ids = [int(re.search(r'id: (\d+)', entry).group(1)) for entry in data_list]
    # Query Milvus for the documents with the specified IDs
    if len(ids)>0:
        res_files = collection.query(
            expr=f"id in {ids}",  # Assuming "prayerId" is the field storing the IDs
            output_fields=["id",'FileName']  # Specify the fields you want to retrieve
        )
        file_names = [i['FileName'] for i in res_files]
        results = collection.query(
        expr=f"FileName in {file_names}",  # Assuming "prayerId" is the field storing the IDs
        output_fields=["id", "orderChunkId","orderText",'FileName','Judge','Casedetails','Date'])
           

        return clean_order(results)
    else:
        return []

def get_subsections():
    vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name = 'section_collection',
    text_field ='sectionText',
    vector_field ='sectionVector',primary_field='id'
)
    template = """"summerise the context data which is Document format , give beutifull string format.

context = '{context}'

Main section = '{question}'

Helpful Answer:"""
    rag_prompt = PromptTemplate.from_template(template)
    retriever =vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    import os
    os.environ['OPENAI_API_KEY'] = "sk-proj-YaiwNp41YbM3VXjGTEM3T3BlbkFJ0f6UtkPwsb97EIxrJBF2"
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4")
    rag_chain = (
        {"context":retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
    )

    return rag_chain