import os
import requests
import asyncio
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec, Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
#from langchain.retrievers import PineconeHybridSearchRetriever  # Adjusted import


# DocumentRetriever Class
class DocumentRetriever:
    def __init__(self, api_key, index_name, directory, hf_api_token):
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
        self.hf_api_token = hf_api_token

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
        all_documents = []
        if os.path.isdir(self.directory):
            loader = DirectoryLoader(self.directory)
            documents = loader.load()
            all_documents.extend(documents)
        elif os.path.isfile(self.directory):
            loader = TextLoader(self.directory)
            documents = loader.load()
            all_documents.extend(documents)
        else:
            raise FileNotFoundError(f"Directory or file not found: {self.directory}")
        
        return all_documents

    def process_documents(self, user_name):
        if not self.documents:
            print("No documents available to process.")
            return None

        if not self.bm25_encoder_fitted:
            texts = [doc.page_content for doc in self.documents]
            if not texts:
                print("No document texts available for BM25 fitting.")
                return None
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
            print(f"Document {doc_index} has {len(chunks)} chunks.")
            
            file_name = os.path.basename(doc.metadata['source'])
            
            chunk_data = []
            for chunk_index, chunk in enumerate(chunks):
                metadata = {"user_name": user_name, "file_name": file_name}
                chunk_data.append((chunk, metadata))
                print(f"Prepared chunk {chunk_index + 1} from document {doc_index}")

            retriever.add_texts([data[0] for data in chunk_data], metadatas=[data[1] for data in chunk_data])

        print("All documents processed and chunks added.")
        return retriever
        

    def query_documents(self, retriever, question, user_name_filter=None):
        if user_name_filter:
            question = f"{question} AND user_name:{user_name_filter}"

        results = retriever.invoke(question)[:3]

        combined_chunks = ""
        for result in results:
            combined_chunks += result.page_content.strip() + " "

        return combined_chunks

    def summarize_combined_chunks(self, combined_chunks):
        headers = {
            "Authorization": f"Bearer {self.hf_api_token}"
        }

        prompt = (
            f"Summarize the following text in 50-75 words, capturing all key details: {combined_chunks}. The summary is: "
        )

        payload = {
            "inputs": prompt,
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers=headers, json=payload
        )

        if response.status_code == 200:
            json_response = response.json()
            if 'generated_text' in json_response[0]:
                summary = json_response[0]['generated_text'].strip()
                start_idx = summary.find("The summary is:")
                if start_idx != -1:
                    summary = summary[start_idx:].strip()
                return summary
            else:
                return "Error: 'generated_text' not found in the response."
        else:
            return f"Error: Received status code {response.status_code}"

# FastAPI Code
app = FastAPI()

html = """


    
        Websocket Demo
        
    
    
    
        # FastAI ChatBot
        ## Get Previous Conversation
        
            
            
            Get Previous Conversation
            Get All Conversations
        
        
        ## Chat
        ## Your ID: 
        
            
            Send
        
        
    
        
        
            var client_id = Date.now()
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket("ws://localhost:8000/ws/" + client_id);
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
            function getPreviousConversation(event) {
                var userId = document.getElementById("userId").value;
                var numRecords = document.getElementById("numRecords").value;
                ws.send("get_previous_conversation " + userId + " " + numRecords);
                event.preventDefault()
            }
            function getAllConversations() {
                var userId = document.getElementById("userId").value;
                ws.send("get_all_conversations " + userId);
            }
        
    

"""

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['chatbot']
collection = db['websockets']

@app.get("/")
async def get():
    return HTMLResponse(html)

# Initialize DocumentRetriever
document_retriever = DocumentRetriever(
    api_key='2b49b06f-ce5b-4726-9fa7-4af7dc7af732', 
    index_name='demo10', 
    directory=r"E:\akshaya\document_upload_2\to_process", 
    hf_api_token='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'
)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.startswith("get_previous_conversation"):
                _, user_id, num_records = data.split()
                user_id = str(user_id)
                num_records = int(num_records)
                
                records = list(collection.find({"user_id": user_id}).limit(num_records))
                messages = "\n".join([record['message'] for record in records])
                
                await manager.send_personal_message(f"Previous Conversation:\n{messages}", websocket)

            elif data.startswith("get_all_conversations"):
                _, user_id = data.split()
                user_id = str(user_id)

                records = list(collection.find({"user_id": user_id}))
                messages = "\n".join([record['message'] for record in records])
                
                await manager.send_personal_message(f"All Conversations:\n{messages}", websocket)

            else:
                # Process the message and store it in the database
                user_name = str(client_id)
                retriever = document_retriever.process_documents(user_name)
                answer = document_retriever.query_documents(retriever, data, user_name)
                summary = document_retriever.summarize_combined_chunks(answer)

                # Save to MongoDB
                collection.insert_one({"user_id": user_name, "message": data})
                await manager.broadcast(f"{user_name}: {data}")
                await manager.send_personal_message(summary, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"User {client_id} disconnected")
