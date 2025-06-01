import streamlit as st
import fitz
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-128k-instruct")
model = AutoModel.from_pretrained("microsoft/phi-3-mini-128k-instruct")
#model = SentenceTransformer("all-MiniLM-L6-v2")

model.eval()

def extract_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def make_chunks(text, chunk_size=512, overlap=50):
    words = text.split()   # <— simple whitespace split
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks

def embed_text(chunks):
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
    # Tokenize and generate embeddings
    with torch.no_grad():

        outputs = model(**inputs)

    #embeddings = model.encode(chunks, convert_to_numpy=True)
    #return embeddings.tolist()

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()

def connect_to_database():
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            uri="milvus_lite.db"
        )
    # return connections.get_connection("default")

def create_collection():
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)

    schema = CollectionSchema(fields=[id_field, vector_field, text_field], description="Document collection")

    collection_name = "Chunked_Doc1"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 128}})
        collection.load()
    else:
        collection = Collection(collection_name)
        collection.load()

    return collection

def upload_to_milvus(collection, uploaded_file):
    text = extract_from_pdf(uploaded_file)
    chunks = make_chunks(text)
    embeddings = embed_text(chunks)
    ids = [f"{uploaded_file.name}-{i}" for i in range(len(chunks))]

    try:
        collection.insert([ids, embeddings, chunks])
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

    
    st.write(f"Total documents in collection: {collection.num_entities}")
    st.success(f"{uploaded_file.name} uploaded and stored successfully!")

def search_database(query: str):
    collection = Collection("Chunked_Doc1")
    query_embedding = embed_text([query])
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    results = collection.search(
        data=query_embedding,
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["text"]
    )

    if results:
        for hits in results:
            for hit in hits:
                st.write(f"Score: {hit.score}")
                st.write(f"Chunk: {hit.entity.get('text')}")
    else:
        st.warning("No results found.")
    
    return results

#Streamlit Interface
st.set_page_config(page_title="RAG Pipeline", layout="wide")
st.title("RAG Pipeline: Upload & Search")

# Tabs
tab1, tab2 = st.tabs(["📤 Upload Document", "🔍 Search Database"])
connect_to_database()

# ---------- Tab 1: Upload ----------
with tab1:
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("Upload File"):
        if uploaded_file:
            collection = create_collection()
            upload_to_milvus(collection, uploaded_file)

            st.success("File uploaded and stored in Milvus.")
            st.write("File type:", uploaded_file.type)
            st.write("File size (bytes):", uploaded_file.size)
        else:
            st.warning("Please select a file before uploading.")

# ---------- Tab 2: Search ----------
with tab2:
    query = st.text_input("Enter your search query:")

    if st.button("Search") and query:
        try:
            results = search_database(query)
            if results and len(results[0]) > 0:
                for i, hit in enumerate(results[0]):
                    st.markdown(f"*Result {i+1}*")
                    st.write(hit.entity.get("text"))
                    st.caption(f"Distance: {hit.distance:.4f}")
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"Search failed: {str(e)}")