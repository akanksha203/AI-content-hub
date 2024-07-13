import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_medical(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text