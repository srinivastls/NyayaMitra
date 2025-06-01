import streamlit as st
import numpy as np
import fitz
from nltk.tokenize import word_tokenize
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, AnnSearchRequest
from transformers import AutoTokenizer, AutoModel
#from sentence_transformers import SentenceTransformer
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

def embed_text1(chunks):
    # Tokenize and generate embeddings
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():

        outputs = model(**inputs)

    #embeddings = model.encode(chunks, convert_to_numpy=True)
    #return embeddings.tolist()

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()

def embed_text(chunks):
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get [batch_size, seq_len, hidden_size]
    embeddings = outputs.last_hidden_state.mean(dim=1)  # (batch_size, hidden_size)

    # Now reduce if needed
    reduced_embeddings = []
    for emb in embeddings:
        reduced = reduce_dim_avgpool(emb.numpy())  # from 3072 -> 384
        reduced_embeddings.append(reduced)

    return reduced_embeddings

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
    schema = CollectionSchema(fields=[id_field, vector_field], description="Document collection")

    collection_name = "Chunked_Docs"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(field_name="embedding", index_params={"index_type": "IVF_FLAT", "params": {"nlist": 128}})
        collection.load()
    else:
        collection = Collection(collection_name)
        collection.load()

    return collection


def reduce_dim_avgpool(embedding_3072):
    vec = np.array(embedding_3072)
    return np.mean(vec.reshape(8, 384), axis=0).tolist()

def upload_to_milvus(collection, uploaded_file):
    text = extract_from_pdf(uploaded_file)
    #print(text)
    chunks = make_chunks(text)
    #print(chunks)
    embeddings = embed_text(chunks)

    ids = [f"{uploaded_file.name}-{i}" for i in range(len(chunks))]

    try:
        collection.insert([ids, embeddings])
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

    
    st.write(f"Total documents in collection: {collection.num_entities}")
    st.success(f"{uploaded_file.name} uploaded and stored successfully!")



def search_database(query: str, limit=5):
    # Embed the search query
    query_embedding = embed_text([query])
    print(query_embedding)
    # Ensure query embedding is not None and has the expected shape
    if query_embedding is None:
        raise ValueError("Query embedding is None, something went wrong during embedding generation.")
    
    

    # Define search parameters
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}

    search_param_1 = {
        "data": [query_embedding],  # Query vector
        "anns_field": "embedding",  # Vector field name
        "param": search_params,     # Search parameters
        "limit": limit,             # Limit the number of results returned
    }

    # Store the request in a list
    req1 = AnnSearchRequest(**search_param_1)
    reqs = [req1]
    print(reqs)

    # Load collection
    collection = Collection(name="Chunked_Docs")
    collection.load()
    query_embedding1=[[-0.05241739,-0.087837964,-0.035892658,0.016270006,0.014959482,-0.053113956,0.052197292,0.010351395,-0.116110854,-0.008159227,-0.058218885,-0.044779886,0.04150063,-0.026458615,-0.0077944947,0.030526843,-0.04667036,-0.015748704,0.029300345,-0.11010819,-0.10619092,-0.056268744,0.0156034855,-0.050534435,0.03224888,0.025752643,-0.033713106,-0.008487782,-0.02179038,-0.07000311,-0.053697497,0.037133265,-0.007121595,0.07580084,-0.011058466,0.09517255,-0.008366606,-0.069496505,0.0458669,-0.022167474,-0.09023898,-0.079806216,0.0048422306,-0.030981917,0.07240255,-0.06830422,-0.061097976,-0.043528203,0.003928899,0.016946789,-0.10361443,-0.05766831,-0.010354616,-0.060256757,-0.07517872,-0.004269325,0.04457515,0.034085207,0.006044248,-0.027365973,-0.0050594434,0.01028053,0.030508434,0.05639067,-0.0046734205,-0.031980403,-0.095240496,0.09099448,0.042207226,-0.02215987,-0.015487126,-0.026200086,-0.06345452,0.09149773,0.10458156,-0.037654232,0.04002322,-0.031715464,0.08839867,0.025058286,-0.009612749,0.014057365,-0.05174387,0.08902434,-0.016347855,-0.05092335,0.004693569,0.049697217,0.03997497,-0.021495752,0.094762705,-0.05442044,0.019193338,0.073463805,-0.039419085,-0.062373344,0.08769716,-0.021512963,-0.02127662,0.03843683,-0.045128398,-0.029358724,-0.022761267,-0.00072754425,-0.069323145,0.07570005,0.08593597,0.030861167,0.047713317,0.008923524,-0.054895017,0.05599894,-0.06101533,-0.046500567,0.041761607,0.061530948,-0.051048297,0.08270016,0.06777304,0.096050814,-0.025927288,0.13971429,-0.059061322,-0.07604662,-0.005291384,-0.024880718,-0.07973313,6.8536625e-33,0.010476713,0.04408853,0.00120931,0.05498639,0.029749066,-0.037171997,0.029895056,0.059876952,-0.104278065,-0.03382345,0.0489642,0.05933405,-0.025742466,0.09437708,0.052347485,-0.004500439,-0.018887293,-0.017624915,-0.0075214617,-0.024605023,0.019343974,-0.030717831,0.008945777,0.010670728,0.04654871,0.01918254,-0.0513182,-0.004777895,0.015367172,0.035943612,0.0052966466,-0.0495867,-0.11119265,-0.014727747,0.019759102,0.04323395,0.029154394,-0.14505872,0.056423537,0.084304415,-0.12143007,0.054565463,0.0125205545,-0.02349925,-0.004835358,0.005818129,0.07519291,0.018020235,0.0425552,0.014518703,-0.11159362,-0.029367484,0.052377302,0.00083623297,-0.008677368,0.053238317,0.033330407,-0.030175064,-0.0007403014,0.015743617,-0.011521357,0.018550564,-0.03831408,-0.051078264,-0.035107955,0.0016127283,0.007804485,-0.0009431147,0.1544057,0.013054286,-0.031449247,0.020268083,0.049477637,0.007124271,-0.04407168,0.021041058,-0.027220277,-0.073158756,0.008004515,0.02470311,0.019882595,0.015131583,0.009435398,-0.07177561,0.049347233,-0.00045162754,0.07163516,-0.07764594,0.006679576,-0.0021747544,-0.006561364,-0.05141759,0.06897952,-0.027955357,-0.09956911,-6.828994e-33,-0.0577656,-0.014301813,-0.0109570315,0.06738248,0.10923928,0.020204283,0.046814885,-0.05041743,0.010799034,0.017436499,0.014944586,0.013145128,-0.010086905,0.013622334,0.013036786,0.028654581,-0.11197697,0.00691549,-0.0945845,0.018617455,0.004562476,0.07668386,-0.03288353,0.04331746,0.04454463,-0.016479691,-0.07324217,-0.05638166,0.015794193,0.03752733,0.01971402,0.009706447,-0.055336684,-0.006526786,0.0775152,-0.030096155,-0.026401749,-0.08184713,-0.005241314,0.079693146,0.056988884,0.017485544,-0.0012957449,0.0019771892,-0.027968274,0.026248565,-0.08318907,0.026320202,0.0020941265,-0.07391526,-0.015756477,-0.031532127,0.009130563,-0.009213953,0.059437435,0.014251392,0.042986065,-0.011515228,-0.020857945,0.013776396,-0.023591653,0.037821498,0.06477113,0.058785796,0.016552296,-0.109316505,-0.08991276,0.115764275,-0.107392766,-0.099468105,-0.0094117755,0.03341425,0.035830118,-0.059474036,-0.1258571,0.0023730842,-0.013781236,0.049972523,-0.016709663,0.03371625,0.027690263,0.01866556,0.053015936,0.00900733,0.01167959,0.10242918,0.048135098,-0.0032435595,-0.0026865362,-0.10325833,-0.038836665,0.03453029,0.016613627,0.008858249,0.0038143054,-4.8794867e-8,0.0080137765,0.040807903,-0.050253104,0.036221977,-0.0010463018,0.009387102,-0.08574998,0.033247236,0.0093909465,0.07266348,0.03905275,-0.052547947,-0.010913439,-0.0067787804,0.08926437,0.004421471,-0.011807894,0.06235917,-0.009902236,-0.061385956,0.08481646,0.062538974,-0.062205568,0.024602247,-0.04610171,0.012240791,-0.008761036,-0.015991226,0.00775843,0.038713843,-0.044340204,-0.03071717,0.01945087,-0.082606085,0.10229535,0.008382972,0.0052968753,-0.0854804,-0.028462827,0.07306979,-0.042448595,-0.0059871054,0.07577977,0.010919886,0.010980405,0.038012873,-0.038580876,-0.00018818172,0.034885373,-0.02719666,-0.0128461,-0.051091995,-0.05874555,0.007501796,0.13325514,0.088371806,-0.04856249,0.025674801,-0.0390092,0.042598218,0.025456635,-0.011997101,-0.012842403,0.032315496]]
    try:
        # Perform the hybrid search
        res = collection.search(
            data=query_embedding1,  # Query vector

            anns_field="embedding",  # Vector field name
            param=search_params,  # Search parameters
              # Search requests
            limit=limit   # Limit the results
        )

        if res is None or len(res) == 0:
            raise ValueError("No search results returned.")
        
    except Exception as e:
        print(f"Search failed: {e}")
        return []

    # Display results
    if res:
        for hits in res:
            for hit in hits:
                st.write(f"Score: {hit.score}")
                st.write(f"Chunk: {hit.entity.get('text')}")
                st.write(f"id: {hit.entity.get('id')}")
                print(hit.entity.get("e"))
    else:
        st.warning("No results found.")

    return res



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

