from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="phi3:mini-128k")
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("/Users/LakshmiSrinivas/Desktop/Final_year/code/data/act_2_of_1956.pdf")

# Load and split the PDF into pages
pages = loader.load_and_split()

# Verify the number of pages loaded
total_pages = len(pages)
print(f"Total pages loaded: {total_pages}")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(pages)

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)

# Create schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
    FieldSchema(name="sectionVector", dtype=DataType.FLOAT_VECTOR, dim=3072), # Vector field for film vectors
    FieldSchema(name="sectionText", dtype=DataType.VARCHAR, max_length=20000)]

schema = CollectionSchema(fields=fields,enable_dynamic_field=True)
# Create collection
collection = Collection(name="section_collection", schema=schema)


# Create index for each vector field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

print("collection created")

collection.create_index("sectionVector", index_params)

print("index created")



from joblib import Parallel, delayed
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Define the function that will process each row
def process_row(idx, text):
    print(idx)
    entity = {
        "sectionVector": embeddings.embed_query(str(text)),
        "sectionText": str(text)
    }
    collection.insert([entity])

# Use joblib's Parallel and delayed to parallelize the loop
# Use threading backend to avoid pickling issues
print("insertion started")
Parallel(n_jobs=-1, backend="threading")(delayed(process_row)(idx, text) for idx, text in enumerate(splits))

print("successfully completed!")
