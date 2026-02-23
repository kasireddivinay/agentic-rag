from pypdf import PdfReader
import os

def load_documents(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())

        elif file.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)

    return documents