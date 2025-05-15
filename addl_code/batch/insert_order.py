from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="phi3:mini-128k")

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect to Milvus
connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)

# Create schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="orderChunkId", dtype=DataType.INT64),
    FieldSchema(name="orderVector", dtype=DataType.FLOAT_VECTOR, dim=3072), # Vector field for film vectors
    FieldSchema(name="orderText", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="FileName", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="Judge", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="Casedetails", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="Date", dtype=DataType.VARCHAR, max_length=200)]

schema = CollectionSchema(fields=fields,enable_dynamic_field=False)
# Create collection
collection = Collection(name="order_collection", schema=schema)


# Create index for each vector field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
print("collection created")

collection.create_index("orderVector", index_params)

print("index created")

import pandas as pd
df = pd.read_excel('/Users/LakshmiSrinivas/Desktop/Final_year/code/data/wa.xlsx')
df = df.fillna('')
df1=df[:1]

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
    splits = splitter.split_text(str(row['Order']))
    results = []
    for i, text in enumerate(splits):
        entity = {
            "orderChunkId": i,
            "orderVector": embeddings.embed_query(str(row["Case Details"])[:190]+" "+ text),
            "orderText": text,
            "FileName": str(row['Filename']),
            "Judge":str(row["Judge"])[:190],
            "Casedetails":str(row["Case Details"])[:190],
            "Date":str(row["Date"])[:190]

        }
        results.append(entity)
    
    chunk_size = 10
    for i in range(0, len(results), chunk_size):
        collection.insert(results[i:i + chunk_size])

# Use joblib's Parallel and delayed to parallelize the loop
# Use threading backend to avoid pickling issues
print("insertion started")
Parallel(n_jobs=-1, backend="threading")(delayed(process_row)(idx, row) for idx, row in df1.iterrows())

print("successfully completed!")
