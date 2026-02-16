from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import os
import pandas as pd

def load_documents(data_dir="data"):
    documents = []

    pdf_dir = os.path.join(data_dir, "pdfs")
    txt_dir = os.path.join(data_dir, "texts")

    if os.path.exists(pdf_dir):
        for file in os.listdir(pdf_dir):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                documents.extend(loader.load())

    if os.path.exists(txt_dir):
        faculty_csv = os.path.join(txt_dir, "all_faculty.csv")
        if os.path.exists(faculty_csv):
            df = pd.read_csv(faculty_csv)
            
            for _, row in df.iterrows():
                name = row['Name']
                designation = row['Designation']
                dept = row['Department']
                specialization = row['Specialization']
                publications = row['Publications']
                profile_url = row.get('Profile_URL', '')
                
                content = f"""{name} is the {designation} of {dept} department at SKCET.
Department: {dept}
Name: {name}
Designation: {designation}
Specialization: {specialization}
Publications: {publications}"""
                
                if profile_url:
                    content += f"\nProfile: {profile_url}"
                
                if "Head of the Department" in designation or "HOD" in designation:
                    content = f"The HOD of {dept} is {name}. " + content
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": "all_faculty.csv",
                        "department": dept,
                        "faculty_name": name,
                        "designation": designation
                    }
                ))
        
        for file in os.listdir(txt_dir):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(txt_dir, file))
                documents.extend(loader.load())

    return documents
