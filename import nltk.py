import os
import re
import nltk
import pandas as pd
import PyPDF2  # For PDF text extraction
import docx2txt  # For DOCX text extraction

RESUME_FOLDER = "Resumes"

def extract_text_from_file(file_path):
    """
    Extracts text from TXT, PDF, or DOCX files.
    """
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n" if page.extract_text() else ""

    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)

    return text.strip() if text else "No text found"

def extract_info(resume_text):
    """
    Extracts name, email, phone number, and skills from a given resume text.
    """
    info = {}

    # Extract email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, resume_text)
    info["email"] = emails[0] if emails else "Not found"

    # Extract phone number
    phone_pattern = r"\+?\d{0,3}[-.\s]?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}"
    phones = re.findall(phone_pattern, resume_text)
    info["phone"] = phones[0] if phones else "Not found"

    # Extract name (First capitalized words at the start of resume)
    name_pattern = r"^\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    name_match = re.search(name_pattern, resume_text)
    info["name"] = name_match.group(1).strip() if name_match else "Not found"

    # Extract skills (Handles multi-line and comma-separated skills)
    skills_pattern = r"(?i)\bskills?\b[:\s]*(.+)"
    skills_match = re.search(skills_pattern, resume_text)
    info["skills"] = skills_match.group(1).strip().lower() if skills_match else "Not found"

    return info

def process_resumes():
    """
    Reads resumes from the folder, extracts info, and saves to a CSV.
    """
    print(f"[INFO] Processing resumes in '{RESUME_FOLDER}' folder...")

    if not os.path.exists(RESUME_FOLDER):
        print("[ERROR] Resumes folder not found!")
        return

    extracted_data = []  # List to store extracted resume data

    for filename in os.listdir(RESUME_FOLDER):
        file_path = os.path.join(RESUME_FOLDER, filename)

        if os.path.isfile(file_path):
            print(f"[INFO] Processing file: {filename}")

            resume_text = extract_text_from_file(file_path)
            extracted_info = extract_info(resume_text)
            extracted_info["filename"] = filename
            extracted_data.append(extracted_info)

            print(f"[RESULT] Extracted Data from {filename}: {extracted_info}\n")

    # Convert list of extracted data to DataFrame
    df = pd.DataFrame(extracted_data)
    df['phone'] = df['phone'].replace("Not found", None)

    # Save to CSV
    csv_file = "extracted_resumes.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"[INFO] All resumes processed! Data saved to {csv_file}")
    print("\n[INFO] Extracted Data in Table Format:")
    print(df.to_string(index=False))  # Ensures a clean table format

# Run the processing function
process_resumes()
