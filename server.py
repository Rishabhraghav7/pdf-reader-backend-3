from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import pandas as pd
import pdfplumber
import fitz  
import re
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to clean text by removing unnecessary blank lines and characters
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\r\t', '', text)   # Remove carriage returns and tabs
    text = text.strip()  # Strip leading/trailing whitespace
    return text

# Function to extract tables using pdfplumber and format them with <table start> and <table end>
def extract_tables(pdf, page_num):
    table_texts = []
    page = pdf.pages[page_num]
    tables = page.extract_tables()
    if tables:
        for table_index, table in enumerate(tables):
            if len(table) > 1:
                table_text = "<table start>\n"
                for row in table:
                    table_text += ",".join([str(cell).replace("\n", " ") for cell in row]) + "\n"
                table_text += "<table end>\n"
                table_texts.append(table_text)
    return table_texts

# Function to parse the table data and convert it into a DataFrame
def parse_table(table_text):
    try:
        csv_content = table_text.split("<table start>\n")[1].split("<table end>\n")[0]
        table_io = io.StringIO(csv_content)
        df = pd.read_csv(table_io, delimiter=",")
        return df
    except Exception as e:
        raise ValueError(f"Error parsing table: {str(e)}")

# Function to summarize tables
def summarize_table(table_data):
    try:
        df = parse_table(table_data)
        if not df.empty:
            return df.describe().to_string()  # Example: returning summary statistics
        else:
            return "Table is empty or could not be parsed."
    except Exception as e:
        return f"Error summarizing table: {str(e)}"

# Endpoint to upload and process PDF files
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file to a temporary location
        temp_pdf_path = f"./temp_{file.filename}"
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_summaries = []

        # Open the PDF once and reuse it across pages
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page_num in range(len(pdf.pages)):
                tables = extract_tables(pdf, page_num)
                for table in tables:
                    table_summary = summarize_table(table)
                    output_summaries.append(table_summary)

        # Return a message if no tables were found
        if not output_summaries:
            return {"status": "No tables found or processed.", "summaries": []}

        return {"status": "success", "summaries": output_summaries}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Clean up the temporary PDF file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
