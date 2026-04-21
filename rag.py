from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = [
    "High debt increases default risk",
    "Late payments indicate high risk",
    "Low income customers are risky",
    "More credit lines increase risk"
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query):
    q = model.encode([query])
    _, idx = index.search(np.array(q), 2)
    return [docs[i] for i in idx[0]]
