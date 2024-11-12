import os
import requests
import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import asyncio
import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pymongo import MongoClient
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec, Pinecone
#from langchain.retrievers import PineconeHybridSearchRetriever  # Adjusted import
from langchain_community.retrievers import PineconeHybridSearchRetriever
 
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
        if self.index_name in self.pc.list_indexes().names():
            self.pc.delete_index(self.index_name)
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
        if len(self.documents) > 0:
            texts = [doc.page_content for doc in self.documents]
            if texts:
                self.bm25_encoder.fit("\n".join(texts))
                self.bm25_encoder.dump("bm25_values.json")
                self.bm25_encoder = BM25Encoder().load("bm25_values.json")

                retriever = PineconeHybridSearchRetriever(
                    embeddings=self.embeddings,
                    sparse_encoder=self.bm25_encoder,
                    index=self.index
                )

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )

                all_chunks = []
                for doc in self.documents:
                    chunks = text_splitter.split_text(doc.page_content)
                    file_name = os.path.basename(doc.metadata['source'])
                    all_chunks.extend((chunk, {"user_name": user_name, "file_name": file_name}) for chunk in chunks)

                retriever.add_texts([data[0] for data in all_chunks], metadatas=[data[1] for data in all_chunks])
                return retriever
        else:
            return None

    def query_documents(self, retriever, question, user_name_filter=None):
        if user_name_filter:
            question = f"{question} AND user_name:{user_name_filter}"
        results = retriever.invoke(question)[:3]  # Limit to top 3 relevant chunks
        combined_chunks = ""
        for result in results:
            combined_chunks += result.page_content.strip() + " "
        return combined_chunks

    def summarize_combined_chunks(self, combined_chunks):
        headers = {
            "Authorization": f"Bearer {self.hf_api_token}"
        }

        prompt = (
            f"Please summarize the following information in 50-75 words, ensuring all key details are captured: {combined_chunks}. The summary should highlight the main achievements and updates while maintaining clarity and coherence.The summary is:"
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
                summary = json_response[0]["generated_text"]
                return summary
            else:
                print("Error: 'generated_text' not found in the response.")
                return "Could not summarize this content."
        else:
            print(f"Error: Received status code {response.status_code}")
            return "Could not summarize this content."

# FastAPI Code
app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Websocket Demo</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
    </head>
    <body>
    <div class="container mt-3">
        <h1>FastAI ChatBot</h1>
        <h2>Get Previous Conversation</h2>
        <form action="" onsubmit="getPreviousConversation(event)">
            <input type="text" class="form-control" id="userId" placeholder="Enter User ID" autocomplete="off"/>
            <input type="number" class="form-control" id="numRecords" placeholder="Enter Number of Records" autocomplete="off"/>
            <button class="btn btn-outline-primary mt-2">Get Previous Conversation</button>
            <button class="btn btn-outline-primary mt-2" onclick="getAllConversations()">Get All Conversations</button>
        </form>
        <div id="conversation-history"></div>
        
        <h2>Chat</h2>
        <h2>Your ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" class="form-control" id="messageText" autocomplete="off"/>
            <button class="btn btn-outline-primary mt-2">Send</button>
        </form>
        <ul id='messages' class="mt-5"></ul>
    </div>
    
        <script>
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
            ws.onopen = function(event) {
                console.log("WebSocket is open now.");
            };
            ws.onclose = function(event) {
                console.log("WebSocket is closed now.");
            };
            ws.onerror = function(event) {
                console.error("WebSocket error observed:", event);
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
        </script>
    </body>
</html>
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

# Define a dictionary of predefined responses
responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi! What's on your mind?",
    "how are you": "I'm doing great, thanks for asking!",
    "your name": "My name is ChatBot",
}

# Initialize DocumentRetriever
document_retriever = DocumentRetriever(
    api_key="2b49b06f-ce5b-4726-9fa7-4af7dc7af732", 
    index_name='demo8', 
    directory=r"C:\Users\risha\OneDrive\Documents\PdfReader\thisisoutput", 
    hf_api_token="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"
)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            user_name = client_id
            if data.startswith("get_previous_conversation"):
                _, user_id, num_records = data.split()
                records = list(collection.find({"user_id": user_id}).limit(int(num_records)))
                history = "\n".join([record["message"] for record in records])
                await manager.send_personal_message(f"Previous conversation: \n{history}", websocket)

            elif data.startswith("get_all_conversations"):
                _, user_id = data.split()
                records = list(collection.find({"user_id": user_id}))
                history = "\n".join([record["message"] for record in records])
                await manager.send_personal_message(f"All conversations: \n{history}", websocket)

            elif data in responses:
                response = responses[data]
                await manager.send_personal_message(response, websocket)

            else:
                retriever = document_retriever.process_documents(user_name)
                combined_chunks = document_retriever.query_documents(retriever, data, user_name)
                summary = document_retriever.summarize_combined_chunks(combined_chunks)
                await manager.send_personal_message(summary, websocket)

            # Save the message to MongoDB
            collection.insert_one({"user_id": user_name, "message": data, "timestamp": datetime.datetime.now()})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"User  {client_id} has disconnected")
