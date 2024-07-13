

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure GenAI with  API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_pro_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text


def process_pdf(file, query, model_path):
    if file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

       
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Load SentenceTransformer model -used to encode sentences into numerical embeddings
        model = SentenceTransformer(model_path)

        # Encode DATA chunks
        embeddings = model.encode(chunks)

        # Encode USER query
        query_embedding = model.encode([query])

        
        similarities = cosine_similarity(query_embedding, embeddings)

        
        top_indices = similarities.argsort(axis=1).flatten()[-5:][::-1]
        retrieved_chunks = [chunks[i] for i in top_indices]

        # Prepare context for Gemini Pro model
        context = " ".join(retrieved_chunks)
        prompt = f"""Based on the following context from the PDF, please answer the question:

        Context: {context}

        Question: {query}

        Answer the question accurately and concisely."""

        
        gemini_response = get_gemini_pro_response(prompt)
        
        return gemini_response
    
    return None
