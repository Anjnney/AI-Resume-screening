"""import os
import re
import nltk
import spacy
import PyPDF2
import numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Extract Named Entities (Skills, Experience, Education)
def extract_named_entities(text):
    doc = nlp(text)
    entities = {"Skills": [], "Experience": [], "Education": []}
    
    # Regex-based skill extraction (improved)
    skill_keywords = {"python", "java", "c++", "sql", "machine learning", "deep learning", "data analysis"}
    for token in doc:
        if token.text.lower() in skill_keywords:
            entities["Skills"].append(token.text)
    
    # Extracting Education & Experience
    for ent in doc.ents:
        if ent.label_ in ["ORG"]:  # Assuming ORG might include universities
            entities["Education"].append(ent.text)
        elif ent.label_ in ["DATE", "TIME"]:  # Experience often has dates
            entities["Experience"].append(ent.text)
    
    return entities

# Convert text data into TF-IDF Vectors
def vectorize_text_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts), vectorizer

# Convert text into Word2Vec Embeddings
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# ML-based Classification Model (Logistic Regression)
def train_ml_classifier(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Function to match resumes to job descriptions
def match_resume_to_job(resume_text, job_desc_text):
    corpus = [resume_text, job_desc_text]
    tfidf_matrix, vectorizer = vectorize_text_tfidf(corpus)
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score

# Example usage
if __name__ == "__main__":
    sample_text = "John has 5 years of experience in Python programming. He graduated from MIT with a Computer Science degree."
    entities = extract_named_entities(sample_text)
    print("Extracted Entities:", entities)
    
    sample_corpus = [sample_text, "Alice has 3 years of experience in data science."]
    tfidf_matrix, feature_names = vectorize_text_tfidf(sample_corpus)
    print("TF-IDF Feature Names:", feature_names)
    
    sentences = [preprocess_text(text) for text in sample_corpus]
    word2vec_model = train_word2vec(sentences)
    print("Word2Vec Model Trained Successfully")
    
    # Matching resume to job description
    job_description = "We are looking for a Python developer with experience in data science."
    match_score = match_resume_to_job(sample_text, job_description)
    print(f"Resume Match Score: {match_score:.2f}")
"""