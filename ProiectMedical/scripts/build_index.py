import os
import sys
from dotenv import load_dotenv

# Ensure the root directory is on the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools.medical_tools import load_corpus
from src.tools.vector_tools import build_index

def main():
    load_dotenv()
    
    # Check if OPENAI_API_KEY is available
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not defined in environment or .env file.")
        sys.exit(1)
        
    persist_dir = "vectorstore"
    
    # Check if index already exists and is not empty to avoid duplicate records
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"WARNING: The directory '{persist_dir}' already exists and is not empty.")
        print("Skipping indexing. If you wish to rebuild the index, delete the directory first.")
        sys.exit(0)
        
    print("Building clinical index...")
    documents = load_corpus("corpus")
    build_index(documents, persist_directory=persist_dir)
    print("Clinical index built successfully!")

if __name__ == "__main__":
    main()
