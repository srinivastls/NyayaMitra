import pandas as pd
df = pd.read_excel('/home/ubuntu/ram/code/notebooks/prayers_data.xlsx')
df = df.fillna('')
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="phi3")
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
# Connect to Milvus
connections.connect(
    host="localhost", # Replace with your Milvus server IP
    port="19530"
)
# Create schema
prayers_fields = [
    FieldSchema(name="prayerId", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="prayerVector", dtype=DataType.FLOAT_VECTOR, dim=3072), # Vector field for film vectors
    FieldSchema(name="prayerText", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="CASE_NO", dtype=DataType.VARCHAR, max_length=400),
    FieldSchema(name="PETITIONER_ADV", dtype=DataType.VARCHAR, max_length=400),
    FieldSchema(name="RESPONDENT_ADV", dtype=DataType.VARCHAR, max_length=400),
    FieldSchema(name="PETITIONER", dtype=DataType.VARCHAR, max_length=400),
    FieldSchema(name="RESPONDENT", dtype=DataType.VARCHAR, max_length=400),
    FieldSchema(name="SUBJECT_CATEGORY", dtype=DataType.VARCHAR, max_length=400)]
prayers_schema = CollectionSchema(fields=prayers_fields,enable_dynamic_field=False)
# Create collection
collection = Collection(name="prayers_collection_updated", schema=prayers_schema)


index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
collection.create_index("prayerVector", index_params)
from joblib import Parallel, delayed
# Define the function that will process each row
def process_row(idx, row):
    print(idx)
    collection.insert( [{"prayerId": idx,
            "prayerVector":embeddings.embed_query(str(row['PRAYER'])), 
            "prayerText": str(row['PRAYER']),
            "CASE_NO":str(row['CASE NO']),
            "PETITIONER_ADV":str(row['PETITIONER ADV.']),
            "RESPONDENT_ADV":str(row['RESPONDENT ADV.']),
            "PETITIONER":str(row['PETITIONER']),
            "RESPONDENT":str(row["RESPONDENT"]),
            "SUBJECT_CATEGORY":str(row['SUBJECT CATEGORY']) 
            }])

# Use joblib's Parallel and delayed to parallelize the loop
Parallel(n_jobs=-1, backend="threading")(delayed(process_row)(idx, row) for idx, row in df[5135:].iterrows())