import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from tqdm import tqdm  # Import tqdm for progress bar

def load_pdfs_and_convert_to_text(folder_path):
    """Convert PDF files in the folder to text and clean them."""
    pdf_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.pdf')]
    
    # Initialize tqdm progress bar
    cleaned_texts = []
    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(folder_path, filename)
        text_content = extract_text_from_pdf(pdf_path)
        cleaned_texts.append(clean_text(text_content))  # Clean while appending
    
    return cleaned_texts


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)


def convert_csvs_to_text(folder_path):
    """Convert CSV files in the folder to text and clean them."""
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]
    
    # Initialize tqdm progress bar
    cleaned_texts = []
    for filename in tqdm(csv_files, desc="Processing CSVs", unit="file"):
        csv_path = os.path.join(folder_path, filename)
        text_content = extract_text_from_csv(csv_path)
        cleaned_texts.append(clean_text(text_content))  # Clean while appending
    
    return cleaned_texts


def extract_text_from_csv(csv_path):
    """Extract text from a CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)


def clean_text(text):
    """Clean the input text by removing extra spaces and formatting."""
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'\n+', '\n', text)        
    return text


def get_cleaned_documents(folder_path):
    """Load and clean all documents from the folder, directly returning cleaned content."""
    pdf_texts = load_pdfs_and_convert_to_text(folder_path)
    csv_texts = convert_csvs_to_text(folder_path)
    all_cleaned_texts = pdf_texts + csv_texts
    return all_cleaned_texts

