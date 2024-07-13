# textgen.py

from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables from .env
load_dotenv()

# configure genai library with api key and load Google Generative AI model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Function to generate content
def generate_content(question):
    response = model.generate_content(question)
    return response.text
