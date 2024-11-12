from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
import os
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pymongo import MongoClient
import asyncio

# DocumentRetriever Class
class DocumentRetriever:
    def __init__(self, api_key, index_name, directory, hf_api_token):  # Corrected constructor
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
            self.bm25_encoder.fit(str(self.documents[0].page_content))
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

            for doc_index, doc in enumerate(self.documents):
                chunks = text_splitter.split_text(doc.page_content)
                file_name = os.path.basename(doc.metadata['source'])
                
                for chunk_index, chunk in enumerate(chunks):
                    retriever.add_texts([chunk], metadatas=[{"user_name": user_name, "file_name": file_name}])

            return retriever
        else:
            return None

    def query_documents(self, retriever, question, user_name_filter=None):
        if user_name_filter:
            question = f"{question} AND user_name:{user_name_filter}"

        # Retrieve more than 3 chunks to get a more complete picture
        results = retriever.invoke(question)[:10]  # Increase to top 10 relevant chunks

        combined_chunks = ""
        for result in results:
            combined_chunks += result.page_content.strip() + " "

        return combined_chunks

    def summarize_combined_chunks(self, combined_chunks):
        headers = {
            "Authorization": f"Bearer {self.hf_api_token}"
        }

        prompt = (
            f"Generate a concise summary capturing the main points from the following text: {combined_chunks}. "
            "The summary should be complete and should not cut off abruptly."
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

                # If the summary ends abruptly, regenerate the summary
                if summary.endswith("and") or summary.endswith(","):
                    print("Incomplete summary detected. Regenerating summary...")
                    return self.summarize_combined_chunks(combined_chunks)  # Recursive call

                # Extract only the desired part of the summary
                start_idx = summary.find("The summary is:")
                if start_idx != -1:
                    summary = summary[start_idx:].strip()  # Keep only the part after "The summary is:"
                return summary
            else:
                return "Error: 'generated_text' not found in the response."
        else:
            return f"Error: Received status code {response.status_code}"

# FastAPI Code
app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Websocket Demo</title>
        <!-- Bootstrap CSS -->
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
        <ul id='messages' class="mt-5">
        </ul>
        
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
    directory=r"C:\\Users\\risha\\OneDrive\\Documents\\PdfReader\\output2", 
    hf_api_token="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"
)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("get_previous_conversation"):
                user_id, num_records = data.split(" ")[1], int(data.split(" ")[2])
                messages = collection.find({"user_id": user_id}).limit(num_records)
                message_history = [message['message'] for message in messages]
                await manager.send_personal_message(f"Previous conversations: {message_history}", websocket)
            elif data.startswith("get_all_conversations"):
                user_id = data.split(" ")[1]
                messages = collection.find({"user_id": user_id})
                all_messages = [message['message'] for message in messages]
                await manager.send_personal_message(f"All conversations: {all_messages}", websocket)
            else:
                await manager.broadcast(f"Client #{client_id} says: {data}")
                collection.insert_one({"user_id": str(client_id), "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")

        