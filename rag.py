from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

docs = [
    "Low income reduces repayment capacity, making it harder to pay EMIs on time.",
    "High debt increases financial burden and leads to higher default probability.",
    "Late payments indicate poor financial discipline and increase credit risk.",
    "Too many credit lines can lead to over-leveraging and repayment stress.",
    "High dependency reduces disposable income, increasing default chances."
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query):
    q = model.encode([query])
    _, idx = index.search(np.array(q), 2)
    return [docs[i] for i in idx[0]]
