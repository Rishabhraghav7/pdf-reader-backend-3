import fitz  # PyMuPDF
import pdfplumber
import os
import csv
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from fastapi import UploadFile
import shutil

# Configuration for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = ''

# Function to clean text by removing unnecessary blank lines and characters
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\r\t', '', text)   # Remove carriage returns and tabs
    text = text.strip()  # Strip leading/trailing whitespace
    return text

# Function to check if a row in a table is empty
def is_row_empty(row):
    return all(cell is None or cell.strip() == '' for cell in row)

# Function to check if two bounding boxes intersect (for text extraction near tables)
def intersect_areas(bbox1, bbox2):
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2
    return not (x1_1 < x0_2 or x0_1 > x1_2 or y1_1 < y0_2 or y0_1 > y1_2)

# Function to determine if an image contains any text (scanned image or not)
def is_scanned_image(image):
    text = pytesseract.image_to_string(image)
    return bool(text.strip())

# Function to extract tables using pdfplumber and format them with <table start> and <table end>
def extract_tables(pdf_path, page_num):
    table_texts = []
    table_areas = []

    with pdfplumber.open(pdf_path) as pdf:
        tables = pdf.pages[page_num].extract_tables()
        if tables:
            for table_index, table in enumerate(tables):
                if len(table) > 1 and not all(is_row_empty(row) for row in table):
                    # Format the table data with <table start> and <table end>
                    table_text = "<table start>\n"
                    for row in table:
                        cleaned_row = [clean_text(cell) for cell in row]
                        table_text += ",".join(f'"{cell}"' if ',' in cell else cell for cell in cleaned_row) + "\n"
                    table_text += "<table end>\n"
                    table_texts.append(table_text)
                    
                    # Get table bounding box to exclude from text extraction
                    table_bbox = pdf.pages[page_num].find_tables()[table_index].bbox
                    table_areas.append(table_bbox)

    return table_texts, table_areas

# Function to extract content from images (OCR for scanned images)
def extract_image_text_from_pdf_page(pdf_path, page_num):
    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
    image_text = ""
    for image in images:
        if is_scanned_image(image):
            image_text = pytesseract.image_to_string(image)
    return image_text

# Main function to process the PDF and generate the report
def extract_contents_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_report = ""

    total_pages = doc.page_count
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        page_number = page_num + 1
        print(f"\n--- Page {page_number} ---")

        # Extract tables with pdfplumber
        table_texts, table_areas = extract_tables(pdf_path, page_num)
        
        # Extract text using PyMuPDF excluding detected table areas
        extracted_text = page.get_text("blocks")
        text = ""
        
        for block in extracted_text:
            bbox = block[:4]  # Get the bounding box of the text block
            block_text = block[4]  # Extract the text from the block

            # Skip text inside table areas
            if not any(intersect_areas(bbox, table_bbox) for table_bbox in table_areas):
                text += block_text + "\n"  # Append block text with a newline

        # If no text is extracted, process the page as an image (OCR)
        if not text.strip():
            print(f"Extracting text via OCR for page {page_number} (image-based PDF)...")
            text = extract_image_text_from_pdf_page(pdf_path, page_num)

        # Clean the combined text
        text = clean_text(text)

        # Combine text and tables in the final report
        full_report += f"\n--- Page {page_number} ---\n{text}\n" + "\n".join(table_texts)

    return full_report


# Function to save the report as a .txt file
def save_report_as_txt(report_text, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(report_text)
    print(f"Text report saved to: {txt_path}")

# FastAPI-specific function to process file uploads
def process_uploaded_pdf(file: UploadFile, output_dir: str):
    # Save the uploaded PDF file to a temporary location
    pdf_path = os.path.join(output_dir, file.filename)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract content from the PDF
    report_content = extract_contents_from_pdf(pdf_path)

    # Save the final report as a .txt file in the output directory
    report_txt_path = os.path.join(output_dir, "output.txt")
    save_report_as_txt(report_content, report_txt_path)

    return report_txt_path, report_content
