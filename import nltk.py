import os
import PyPDF2
from docx import Document
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# Function to clean and preprocess text (tokenization, stopwords removal, lemmatization)
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to structure the extracted data
def process_resume(resume_path):
    file_extension = os.path.splitext(resume_path)[1].lower()
    
    # Extract text based on file type (PDF or DOCX)
    if file_extension == '.pdf':
        text = extract_text_from_pdf(resume_path)
    elif file_extension == '.docx':
        text = extract_text_from_docx(resume_path)
    else:
        return f"Unsupported file format: {file_extension}"

    # Preprocess the extracted text
    cleaned_text = preprocess_text(text)
    
    # Structure the data (sample structure - Modify as needed)
    result = {
        'Name': extract_name_from_text(text),  # You'll need to create an extraction function for name
        'Email': extract_email_from_text(text),  # Email extraction function
        'Phone': extract_phone_from_text(text),  # Phone number extraction function
        'Extracted Text Preview': cleaned_text[:200]  # Preview first 200 characters
    }
    
    return result

# Sample name extraction function (This can be improved for more accuracy)
def extract_name_from_text(text):
    # A simple heuristic to grab a name (you may want to improve this based on format)
    name_pattern = r"([A-Z][a-z]+\s[A-Z][a-z]+)"
    match = re.search(name_pattern, text)
    return match.group(0) if match else "Name not found"

# Email extraction using regex
def extract_email_from_text(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group(0) if match else "Email not found"

# Phone number extraction using regex (simple pattern for US/Intl format)
def extract_phone_from_text(text):
    phone_pattern = r"\+?[0-9]{1,4}?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}"
    match = re.search(phone_pattern, text)
    return match.group(0) if match else "Phone not found"

# Example usage
if __name__ == "__main__":
    resume_folder = "C:/Users/anjnn/OneDrive/Desktop/KJSCE/SEM 4/AI Resume Project/Resumes" # Path to the folder containing resumes
    resumes = os.listdir(resume_folder)
    
    results = []  # Create an empty list to store the results

    # Process each resume in the folder
    for resume in resumes:
        resume_path = os.path.join(resume_folder, resume)
        result = process_resume(resume_path)
        print(f"[INFO] Processed file: {resume}")
        results.append(result)  # Store the result in the results list

    # Now loop through the collected results
    for result in results:
        # Check if result is a dictionary
        if isinstance(result, dict):
            print(f"[RESULT] Extracted:\n  Name: {result.get('Name', 'Not found')}\n  Email: {result.get('Email', 'Not found')}\n  Phone: {result.get('Phone', 'Not found')}\n  Text Preview: {result.get('Extracted Text Preview', 'Not found')}\n")
        else:
            print("[ERROR] Result is not in the expected format:", result)
