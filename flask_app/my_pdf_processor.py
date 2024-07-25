# my_pdf_processor.py
import logging
from pdfminer.high_level import extract_text
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return None

def process_pdf_query(pdf_path, query):
    try:
        text = read_pdf(pdf_path)
        if text is None:
            return "Failed to read the PDF file."

        # Load the question-answering pipeline
        nlp = pipeline("question-answering")

        # Prepare the input for the QA model
        qa_input = {
            'question': query,
            'context': text
        }

        # Get the answer
        response = nlp(qa_input)
        return response['answer']

    except Exception as e:
        logger.error(f"Error processing PDF query: {e}")
        return "An error occurred while processing the query."