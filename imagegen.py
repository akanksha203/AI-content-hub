# textgen.py
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import os


load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

# Function to generate content
def gen(input_text, image_path):
    image = Image.open(image_path)
    if input_text:
        response = model.generate_content([image, input_text])
    else:
        response = model.generate_content(image)
    return response.text
