from google.cloud import storage, firestore
from PyPDF2 import PdfReader
import openai
import os
import tempfile
from langdetect import detect
from googletrans import Translator

# Initialize clients
storage_client = storage.Client()
firestore_client = firestore.Client()
translator = Translator()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Configuration
DOCUMENT_BUCKET = "your-document-bucket"
PROMPT_COLLECTION = "document_prompts"
CHUNK_SIZE = 2000  # tokens
OVERLAP = 100      # tokens

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def process_document(document_keyword):
    """Process and cache documents for a specific keyword"""
    docs_ref = firestore_client.collection(PROMPT_COLLECTION)
    query_ref = docs_ref.where("keywords", "array_contains", document_keyword)
    documents = [doc.to_dict() for doc in query_ref.stream()]
    
    processed_data = []
    
    for doc in documents:
        # Download document from GCS
        bucket = storage_client.bucket(DOCUMENT_BUCKET)
        blob = bucket.blob(doc["gcs_path"])
        
        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_filename(temp_file.name)
            
            # Extract text
            if doc["gcs_path"].endswith(".pdf"):
                reader = PdfReader(temp_file.name)
                text = "\n".join([page.extract_text() for page in reader.pages])
            else:
                with open(temp_file.name, "r") as f:
                    text = f.read()
        
        # Store processed data
        processed_data.append({
            "text": text,
            "prompt_examples": doc.get("examples", []),
            "metadata": doc
        })
    
    return processed_data

def construct_prompt(query, language, document_data):
    """Build the LLM prompt with examples and context"""
    system_prompt = """You are a policy analysis assistant. Answer questions using the provided context and examples.
    Current policy focus: {keyword}
    Answer in {language} language.
    
    Examples of good answers:
    {examples}
    
    Context:
    {context}
    """
    
    # Combine all examples
    examples = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" 
                        for doc in document_data 
                        for ex in doc['prompt_examples']])
    
    # Combine all text chunks
    context = "\n\n".join([doc['text'] for doc in document_data])
    
    return system_prompt.format(
        keyword=document_keyword,
        language=language,
        examples=examples,
        context=context
    )

def policy_query_handler(request):
    """HTTP Cloud Function"""
    request_json = request.get_json()
    query = request_json.get("query")
    document_keyword = request_json.get("keyword", "DPDPA")  # Default keyword
    user_language = request_json.get("lang", "en")
    
    try:
        # Detect language if not specified
        if user_language == "auto":
            user_language = detect(query)
        
        # Translate non-English queries
        if user_language != "en":
            translated_query = translator.translate(query, dest="en").text
        else:
            translated_query = query
        
        # Process relevant documents
        document_data = process_document(document_keyword)
        
        # Construct LLM prompt
        prompt = construct_prompt(translated_query, user_language, document_data)
        
        # Generate response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": translated_query}
            ],
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": [doc["metadata"] for doc in document_data]
        }
    
    except Exception as e:
        return {"error": str(e)}, 500

# --------------------------
# Document Processing Trigger
# --------------------------

def process_new_document(event, context):
    """Background function to process uploaded documents"""
    file_name = event['name']
    bucket_name = event['bucket']
    
    # Get document metadata
    doc_ref = firestore_client.collection(PROMPT_COLLECTION).document(file_name)
    metadata = doc_ref.get().to_dict()
    
    # Only process documents with metadata
    if metadata:
        # Download and process document
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_filename(temp_file.name)
            
            if file_name.endswith(".pdf"):
                reader = PdfReader(temp_file.name)
                text = "\n".join([page.extract_text() for page in reader.pages])
            else:
                with open(temp_file.name, "r") as f:
                    text = f.read()
        
        # Update metadata with processed text
        doc_ref.update({
            "processed_text": text,
            "processed_at": firestore.SERVER_TIMESTAMP
        })