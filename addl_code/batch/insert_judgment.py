from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="phi3")

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)
 
# Create index for each vector field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="judgementChunkId", dtype=DataType.INT64),
    FieldSchema(name="judgementVector", dtype=DataType.FLOAT_VECTOR, dim=3072),  # Adjust the dimension if needed
    FieldSchema(name="judgementText", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="judgementFileName", dtype=DataType.VARCHAR, max_length=255)
]

# Create the collection schema
collection_schema = CollectionSchema(fields=fields, description="Collection for storing judgement data")

# Create the collection
collection_name = "JudgementCollection"  # Replace with your desired collection name
collection = Collection(name=collection_name, schema=collection_schema)

print("collection created")

collection.create_index("judgementVector", index_params)

print("index created")

import pandas as pd
df1 = pd.read_csv('/Users/LakshmiSrinivas/Desktop/Final_year/code/data/cleaned_combined_output.csv')
df=df1[:10]
print("dataframe loaded successfully")

from joblib import Parallel, delayed
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the objects outside the function
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,  # Maximum size of each chunk
    chunk_overlap=50  # Overlap between chunks to maintain context
)

# Define the function that will process each row
def process_row(idx, row):
    print(idx)
    splits = splitter.split_text(str(row['judgment']))
    results = []
    for i, text in enumerate(splits):
        entity = {
            "judgementChunkId": i,
            "judgementVector": embeddings.embed_query(text),
            "judgementText": text,
            "judgementFileName": row['file_name']
        }
        results.append(entity)
    
    chunk_size = 10
    for i in range(0, len(results), chunk_size):
        collection.insert(results[i:i + chunk_size])

# Use joblib's Parallel and delayed to parallelize the loop
# Use threading backend to avoid pickling issues
print("insertion started")
Parallel(n_jobs=-1, backend="threading")(delayed(process_row)(idx, row) for idx, row in df.iterrows())

print("successfully completed!")
