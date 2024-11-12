from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import DirectoryLoader
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
import os
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentRetriever:
    def __init__(self, api_key, index_name, directory):
        self.pc = Pinecone(api_key=api_key)
        self.cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        self.region = os.environ.get('PINECONE_REGION') or 'us-east-1'
        self.spec = ServerlessSpec(cloud=self.cloud, region=self.region)
        self.index_name = index_name
        self.directory = directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.bm25_encoder = BM25Encoder()
        self.index = self._initialize_index()
        self.documents = self.load_docs()
        self.bm25_encoder_fitted = False  

    def _initialize_index(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                self.index_name,
                dimension=384,
                metric='dotproduct',
                spec=self.spec
            )
        return self.pc.Index(self.index_name)

    def load_docs(self):
        loader = DirectoryLoader(self.directory)
        documents = loader.load()
        if len(documents) == 0:
            raise ValueError("No documents found in the specified directory.")
        return documents

    def process_documents(self, user_name):
        if not self.documents:
            raise ValueError("No documents available to process.")

        if not self.bm25_encoder_fitted:
            texts = [doc.page_content for doc in self.documents]
            self.bm25_encoder.fit("\n".join(texts))
            self.bm25_encoder.dump("bm25_values.json")
            self.bm25_encoder_fitted = True

        retriever = PineconeHybridSearchRetriever(
            embeddings=self.embeddings,
            sparse_encoder=self.bm25_encoder,
            index=self.index
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        for doc_index, doc in enumerate(self.documents):
            chunks = text_splitter.split_text(doc.page_content)
            file_name = os.path.basename(doc.metadata['source'])
            chunk_data = []
            for chunk_index, chunk in enumerate(chunks):
                metadata = {"user_name": user_name, "file_name": file_name}
                chunk_data.append((chunk, metadata))

            retriever.add_texts([data[0] for data in chunk_data], metadatas=[data[1] for data in chunk_data])

        return retriever

    def query_documents(self, retriever, question, user_name_filter=None):
        if user_name_filter:
            question = f"{question} AND user_name:{user_name_filter}"

        results = retriever.invoke(question)
        return results

# Data models for FastAPI requests
class QueryRequest(BaseModel):
    name: str
    question: str
    user_name_filter: str = None

retriever_instance = DocumentRetriever(api_key="8da88d83-6107-45d7-b238-520421d70179", index_name='demo10', directory=r"C:\Users\risha\OneDrive\Documents\PdfReader\output3")

@app.post("/process/")
async def process_documents(request: QueryRequest):
    try:
        retriever = retriever_instance.process_documents(request.name)
        return {"message": "Documents processed successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query/")
async def query_documents(request: QueryRequest):
    try:
        retriever = retriever_instance.process_documents(request.name)  
        results = retriever_instance.query_documents(retriever, request.question, request.user_name_filter)
        return {"results": [result.page_content.strip() for result in results]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
