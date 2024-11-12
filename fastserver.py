from fastapi import FastAPI, UploadFile, File
import os
from Extract_from_pdf import process_uploaded_pdf
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your React app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint to upload a PDF and process its contents
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        output_dir = "output5"  # Directory to save the output
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        
        # Process the uploaded PDF file
        report_txt_path, report_content = process_uploaded_pdf(file, output_dir)

        return {
            "status": "PDF processed successfully",
            "output_path": report_txt_path,
            "report_content": report_content[:1000]  # Return first 1000 chars for preview
        }

    except Exception as e:
        return {"error": str(e)}

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, reload=True)