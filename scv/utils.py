import os
import requests
import tarfile
import glob
from pypdf import PdfReader 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_tar_gz(file_path, extract_path):
    try:
        if tarfile.is_tarfile(file_path):
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            return True
        return False
    except Exception:
        return False

def get_text_from_latex(directory):
    text_content = []
    tex_files = glob.glob(os.path.join(directory, '**/*.tex'), recursive=True)
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                text_content.append(f"--- File: {os.path.basename(tex_file)} ---\n")
                text_content.append(f.read())
                text_content.append("\n")
        except:
            pass
    return "\n".join(text_content)

def get_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception:
        return ""
