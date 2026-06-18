from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vs = Chroma(persist_directory="chroma_db/", embedding_function=embeddings)

results = vs._collection.get(limit=5)
for meta in results["metadatas"]:
    print(meta)
    