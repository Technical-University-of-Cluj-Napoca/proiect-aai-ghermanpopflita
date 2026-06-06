import os
from src.dtos import DocumentType

def determine_doc_type(filepath: str) -> DocumentType:
    normalized = filepath.replace("\\", "/").lower()
    if "pubmed" in normalized:
        return DocumentType.PUBMED_ABSTRACT
    elif "who_guidelines" in normalized:
        if "esc_" in normalized or "ean_" in normalized:
            return DocumentType.PROTOCOL_SPITAL
        return DocumentType.GHID_CLINIC
    elif "icd11" in normalized:
        return DocumentType.ICD11
    elif "national_protocols" in normalized:
        return DocumentType.PROTOCOL_OMS
    elif "hospital_protocols" in normalized:
        return DocumentType.PROTOCOL_SPITAL
    return DocumentType.GHID_CLINIC

def parse_metadata_from_content(content: str):
    # Attempt to parse source, evidence level, etc. from text content if present
    evidence_level = "C" # Default
    source = "Unknown"
    
    lines = content.strip().split("\n")
    for line in lines[:5]: # Search in first 5 lines
        if line.lower().startswith("pmid:"):
            source = "pubmed_" + line.split(":", 1)[1].strip()
        elif line.lower().startswith("code:"):
            source = line.split(":", 1)[1].strip()
        elif line.lower().startswith("document code:"):
            source = line.split(":", 1)[1].strip()
            
    for line in lines[-3:]: # Search in last 3 lines
        if "evidence level:" in line.lower():
            evidence_level = line.lower().split("evidence level:", 1)[1].strip().upper()
            
    return source, evidence_level

def load_corpus(corpus_dir: str):
    documents = []
    print(f"Traversing corpus directory: {corpus_dir}")
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    doc_type = determine_doc_type(filepath)
                    source, evidence_level = parse_metadata_from_content(content)
                    
                    # If it's pubmed or similar we might have set the source in the file
                    # Let's fallback to filename if we couldn't parse
                    if source == "Unknown":
                        source = os.path.splitext(file)[0]
                        
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": source,
                            "doc_type": doc_type.value,
                            "evidence_level": evidence_level,
                            "filepath": filepath
                        }
                    })
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    print(f"Loaded {len(documents)} documents from corpus.")
    return documents
