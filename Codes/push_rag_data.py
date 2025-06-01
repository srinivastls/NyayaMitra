from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import requests
import torch
import json
import torch.nn.functional as F

def make_chunks(text, chunk_size=400, overlap=50):
    words = text.split()   
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
        if i + chunk_size >= len(words):
            break
    return chunks

def embed_text(chunks, tokenizer, model):
    # Add 'query: ' prefix if embedding for queries (optional for docs)
    chunks = ["query: " + text for text in chunks]

    # Tokenize input
    inputs = tokenizer(
        chunks,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

    # Attention-aware mean pooling
    attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
    masked_hidden = hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    # Normalize embeddings (recommended for cosine similarity)
    normalized_embeddings = F.normalize(mean_pooled, p=2, dim=1)

    # print("Embedding shape:", len(normalized_embeddings), "x", len(normalized_embeddings[0]))

    return normalized_embeddings.cpu().numpy().tolist()

embedding_model_name = "BAAI/bge-large-en"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModelForCausalLM.from_pretrained(embedding_model_name)
embedding_model.eval()

df1 = pd.read_csv('cleaned_combined_output.csv')
df=df1[25000:]
url = "https://in03-505c80f9dc0263a.serverless.gcp-us-west1.cloud.zilliz.com/v2/vectordb/entities/insert"
headers = {
  "Authorization": "Bearer 1831fe961113fd3aa8af9c99b8a7d2f3e87b34bc2ac4d0f5bc701b1b09049be2aaba097631a7c8943701cb39b7f625597eb16bae",
  "Accept": "application/json",
  "Content-Type": "application/json"
}

for index, row in df.iterrows():
    chunks = make_chunks(str(row['judgment']))
    metadata = row["meta_data"]
    embedding = embed_text(chunks, embedding_tokenizer, embedding_model)
    # print("x: ", len(row['judgment']))
    # print(len(chunks))
    for i, chunk in enumerate(chunks):
        pk = row['file_name'] + "-" + str(i)
        filename = row['file_name']

        payload = {
            "collectionName": "RAG_legal",
            "data": [
                {
                    "primary_key": pk,
                    "text": chunk,
                    "filename": filename,
                    "embedding": embedding[i],
                    "metadata": metadata
                }
            ]
        }

        response = requests.post(url, data=json.dumps(payload), headers=headers)

        if response.status_code == 200:
            print(f"Inserted: {pk}")
        else:
            print(f"Failed ({response.status_code}) - {pk}")
            print(response.text)