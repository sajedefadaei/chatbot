import os
import chromadb
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

loader = DirectoryLoader("./pubmed_papers", glob="*.txt",loader_cls=TextLoader)
texts = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
docs = text_splitter.split_documents(texts)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_name = "NeuML/biomedbert-base-embeddings"
model_kwargs = {"device": device}
huggingface_ef = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)

db_location = "./chromadb"
client = chromadb.PersistentClient(db_location)

langchain_vectorstore = Chroma(
    client=client,
    collection_name="Info_variations",
    embedding_function=huggingface_ef
)

if __name__ == "__main__":
    langchain_vectorstore.add_documents(
        documents=docs
    )

retriever = langchain_vectorstore.as_retriever(search_kwargs={"k": 5})