import os
import re
import nltk
import spacy
import streamlit as st
from docx import Document
from pypdf import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text if text else "No text extracted from PDF."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def preprocess_text(text):
    tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

def extract_name(text):
    match = re.search(r"([A-Z][a-z]+\s[A-Z][a-z]+)", text)
    return match.group(0) if match else "Name not found"

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Email not found"

def extract_phone(text):
    match = re.search(r"\+?[0-9]{1,4}?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}", text)
    return match.group(0) if match else "Phone not found"

def extract_education(text):
    match = re.search(r"(B\.Tech|M\.Tech|Bachelor's|Master's|Ph\.D|BSc|MSc|MBA)", text)
    return match.group(0) if match else "Education not found"

def vectorize_text(text_data):
    return TfidfVectorizer(stop_words='english').fit_transform(text_data)

def rank_resume(candidate_text, ideal_candidate_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([candidate_text, ideal_candidate_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)

def process_resume(text):
    cleaned_text = preprocess_text(text)
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Sample ideal candidate profile (modify as needed)
    ideal_candidate = "B.Tech degree in Computer Science with 3+ years experience in AI and Data Science. Strong Python and Machine Learning skills."
    ranking_score = rank_resume(cleaned_text, ideal_candidate)
    
    return {
        'Name': extract_name(text),
        'Email': extract_email(text),
        'Phone': extract_phone(text),
        'Education': extract_education(text),
        'Extracted Text Preview': cleaned_text[:200],
        'Named Entities': entities,
        'Resume Match Score (%)': ranking_score
    }

# Streamlit UI
st.title("AI Resume Screening System")
uploaded_file = st.file_uploader("Upload a Resume (DOCX or PDF)", type=["docx", "pdf"])

if uploaded_file:
    text = ""
    if uploaded_file.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    
    result = process_resume(text)
    
    st.subheader("Extracted Resume Information:")
    for key, value in result.items():
        st.write(f"{key}: {value}")