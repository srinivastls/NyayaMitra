from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="phi3")
from langchain_community.document_loaders import PyPDFLoader
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)


# Create collection
collection = Collection(name="section_collection")



from joblib import Parallel, delayed
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

with open('/home/ubuntu/ram/code/pdf_proess/output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


# Define the function that will process each row
def process_row(key , item):
    print(key)
    if len(item) > 0:
        item_strig = " ".join(item)
        document = Document(item_strig)
        splits = text_splitter.split_documents([document])
        for text in splits:
            entity = {
                "sectionVector": embeddings.embed_query(str(key)+":"+str(text)),
                "sectionText": str(key)+":"+str(text)
            }
            collection.insert([entity])


# Use joblib's Parallel and delayed to parallelize the loop
# Use threading backend to avoid pickling issues
print("insertion started")
Parallel(n_jobs=-1, backend="threading")(delayed(process_row)(key, item) for key , item in data.items())

print("successfully completed!")
