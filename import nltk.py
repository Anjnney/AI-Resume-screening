import os
import re
from docx import Document
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

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

# Function to extract relevant details from resumes
def process_resume(resume_path):
    file_extension = os.path.splitext(resume_path)[1].lower()

    # Extract text based on file type (DOCX)
    if file_extension == '.docx':
        text = extract_text_from_docx(resume_path)
    else:
        return f"Unsupported file format: {file_extension}"

    # Preprocess the extracted text
    cleaned_text = preprocess_text(text)
    
    # Extract Named Entities (NER)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # TF-IDF Vectorization
    tfidf_vectorized = vectorize_text([cleaned_text])

    # Structure the data
    result = {
        'Name': extract_name_from_text(text),
        'Email': extract_email_from_text(text),
        'Phone': extract_phone_from_text(text),
        'Education': extract_education_from_text(text),
        'Extracted Text Preview': cleaned_text[:200],  # Preview first 200 characters
        'Named Entities': entities,  # Extracted named entities
        'TF-IDF Vectors': tfidf_vectorized.toarray()  # TF-IDF Vectors for further use
    }
    
    return result

# Function to extract relevant details from job descriptions
def process_job_description(job_desc_path):
    file_extension = os.path.splitext(job_desc_path)[1].lower()

    # Extract text based on file type (TXT)
    if file_extension == '.txt':
        with open(job_desc_path, 'r') as file:
            text = file.read()
    else:
        return f"Unsupported file format: {file_extension}"

    # Preprocess the extracted text
    cleaned_text = preprocess_text(text)
    
    # Extract Named Entities (NER) from Job Description
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # TF-IDF Vectorization
    tfidf_vectorized = vectorize_text([cleaned_text])

    # Structure the data
    result = {
        'Job Title': extract_job_title_from_text(text),
        'Required Skills': extract_skills_from_text(text),
        'Experience Requirements': extract_experience_from_text(text),
        'Text Preview': cleaned_text[:200],  # Preview first 200 characters
        'Named Entities': entities,  # Extracted named entities
        'TF-IDF Vectors': tfidf_vectorized.toarray()  # TF-IDF Vectors for further use
    }
    
    return result

# Sample name extraction function (This can be improved for more accuracy)
def extract_name_from_text(text):
    name_pattern = r"([A-Z][a-z]+\s[A-Z][a-z]+)"
    match = re.search(name_pattern, text)
    return match.group(0) if match else "Name not found"

# Job title extraction function (Basic example, modify based on format)
def extract_job_title_from_text(text):
    title_pattern = r"(Software Engineer|Data Scientist|Accountant)"  # Modify as needed
    match = re.search(title_pattern, text)
    return match.group(0) if match else "Job Title not found"

# Email extraction using regex
def extract_email_from_text(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group(0) if match else "Email not found"

# Skills extraction using regex (Modify to match skills in job description)
def extract_skills_from_text(text):
    skills_pattern = r"(Python|Java|SQL|C\+\+)"  # Example skills, extend as needed
    skills = re.findall(skills_pattern, text)
    return ', '.join(skills) if skills else "Skills not found"

# Experience extraction (e.g., 3+ years, 5 years of experience)
def extract_experience_from_text(text):
    experience_pattern = r"(\d+\+? years? of experience)"
    match = re.search(experience_pattern, text)
    return match.group(0) if match else "Experience not found"

# Education extraction (e.g., B.Tech, Master's)
def extract_education_from_text(text):
    education_pattern = r"(B\.Tech|M\.Tech|Bachelor's|Master's|Ph\.D)"
    match = re.search(education_pattern, text)
    return match.group(0) if match else "Education not found"

# Phone number extraction using regex (simple pattern for US/Intl format)
def extract_phone_from_text(text):
    phone_pattern = r"\+?[0-9]{1,4}?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}"
    match = re.search(phone_pattern, text)
    return match.group(0) if match else "Phone not found"

# Function for TF-IDF Vectorization (convert text data into vectors)
def vectorize_text(text_data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix

# Folder path for job descriptions
job_desc_folder = r"C:/Users/anjnn/OneDrive/Desktop/KJSCE/SEM 4/AI Resume Project/job_desc_folder"

# Debugging step: print the folder path
print(f"Job Descriptions Folder Path: {job_desc_folder}")

# Check if the path exists
if not os.path.exists(job_desc_folder):
    print(f"Error: The folder '{job_desc_folder}' does not exist.")
else:
    job_descriptions = os.listdir(job_desc_folder)
    for job_desc in job_descriptions:
        job_desc_path = os.path.join(job_desc_folder, job_desc)
        print(f"[INFO] Processing: {job_desc_path}")
        result = process_job_description(job_desc_path)
        print(f"[INFO] Processed file: {job_desc}")
        print(f"[RESULT] Extracted:\n  Job Title: {result['Job Title']}\n  Required Skills: {result['Required Skills']}\n  Experience Requirements: {result['Experience Requirements']}\n  Text Preview: {result['Text Preview']}\n  TF-IDF Vectors: {result['TF-IDF Vectors'][:2]}")  # Preview the first 2 values of TF-IDF vector

#Streamlit interface

st.title("AI Resume Screening System")
uploaded_file = st.file_uploader("Upload a Resume (DOCX)", type ="docx")

if uploaded_file is not None:
    # save file temporarily
    file_path = os.path.join("Uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # process the resume
    result = process_resume(file_path)

    #Display the processed resume details

    st.subheader("Extracted Resume Information:")
    st.write(f"**Name:** {result['Name']}")
    st.write(f"**Email:** {result['Email']}")
    st.write(f"**Phone:** {result['Phone']}")
    st.write(f"**Education**{result['Education']}")
    st.write(f"**Extracted Text Preview:** {result['Extracted Text Preview']}")

    st.subheader("Named Entities Extracted:")
    st.write(result['Named Entities'])

    st.subheader("TF-IDF Vectors Preview:")
    st.write(result['TF-IDF Vectors'][:2])  # Preview first 2 TF-IDF vectors