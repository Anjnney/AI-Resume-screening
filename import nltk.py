import os
import re
import nltk
import pandas as pd

"""nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK dependencies are installed!")
except LookupError:
    print("❌ NLTK dependencies missing! Run: nltk.download('punkt')") """


# Folder containing resumes
RESUME_FOLDER = "Resumes"

def extract_info(resume_text):
    """
    Extracts name, email, phone number, and skills from a given resume text.
    """
    print("[INFO] Extracting information from resume...")

    info = {}

    # Extract email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = re.findall(email_pattern, resume_text)
    info["email"] = emails[0] if emails else "Not found"
    print(f"[DEBUG] Extracted Email: {info['email']}")

    # Extract phone number (Handles multiple formats)
    phone_pattern = r"\+?\d{0,3}[-.\s]?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}"
    phones = re.findall(phone_pattern, resume_text)
    info["phone"] = phones[0] if phones else "Not found"
    print(f"[DEBUG] Extracted Phone Number: {info['phone']}")

    # Extract name (First capitalized words at the start of resume)
    name_pattern = r"^\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    name_match = re.search(name_pattern, resume_text)
    info["name"] = name_match.group(1).strip() if name_match else "Not found"
    print(f"[DEBUG] Extracted Name: {info['name']}")

    # Extract skills (Handles multi-line and comma-separated skills)
    skills_pattern = r"(?i)\bskills?\b[:\s]*(.+)"
    skills_match = re.search(skills_pattern, resume_text)
    info["skills"] = skills_match.group(1).strip() if skills_match else "Not found"
    print(f"[DEBUG] Extracted Skills: {info['skills']}")

    print("[INFO] Extraction completed!\n")
    return info



# ===========================================
# 📝 [WEEK 1] Resume Processing Function
# ===========================================
# This function was implemented in Week 1 to process all resumes in the "Resumes" folder.
# It reads resume files, extracts key information, and displays results.
"""def process_resumes():
    
Reads all resumes from the folder and extracts information.
    
    print(f"[INFO] Processing resumes in '{RESUME_FOLDER}' folder...")

    if not os.path.exists(RESUME_FOLDER):
        print("[ERROR] Resumes folder not found!")
        return

    for filename in os.listdir(RESUME_FOLDER):
        file_path = os.path.join(RESUME_FOLDER, filename)

        if os.path.isfile(file_path):
            print(f"[INFO] Processing file: {filename}")

            with open(file_path, "r", encoding="utf-8") as file:
                resume_text = file.read()

            extracted_info = extract_info(resume_text)
            print(f"[RESULT] Extracted Data from {filename}: {extracted_info}\n")
    
    print("[INFO] All resumes processed!")

"""

def process_resumes():
    """
    Reads all resumes from the folder, extracts information, and saves it to a CSV file.
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

            with open(file_path, "r", encoding="utf-8") as file:
                resume_text = file.read()

            extracted_info = extract_info(resume_text)
            extracted_info["filename"] = filename  # Add filename for reference
            extracted_data.append(extracted_info)

            print(f"[RESULT] Extracted Data from {filename}: {extracted_info}\n")

    # Convert list of extracted data to DataFrame
    df = pd.DataFrame(extracted_data)


    # clean data: convert skills to lowercase
    df['skills'] = df['skills'].str.lower()

    #Replace "Not found" phone numbers with None
    df['phone'] = df['phone'].replace("Not found", None)

    # Save to CSV
    csv_file = "extracted_resumes.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"[INFO] All resumes processed! Data saved to {csv_file}")
    print("\n[INFO] Extracted Data in Table Format:")
    print(df.to_string(index=False))  # Ensures a clean table format



# Run the processing function
process_resumes()


