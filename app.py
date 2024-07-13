from flask import Flask, render_template, request, jsonify, url_for
import os
from textsumm import summarizer
from abstext import summarizer2
from textgen import generate_content
from medical import get_medical
from imagegen import gen
from askpdf import process_pdf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text_summarizer')
def text_summarizer():
    return render_template('text_summarizer.html')

@app.route('/analyze',methods=['GET','POST'])
def analyze():
    if request.method=='POST':
        rawtext=request.form['rawtext']
        summarize_type = request.form['summarize_type']

        if summarize_type == 'extractive':
            summary,original_txt,len_orig_txt,len_summary=summarizer(rawtext)
            return render_template('summary.html',summary=summary,original_txt=original_txt,len_orig_txt=len_orig_txt,len_summary=len_summary)
        elif summarize_type=='abstractive':
            summary,original_txt,len_orig_txt,len_summary=summarizer2(rawtext)
            return render_template('summary.html',summary=summary,original_txt=original_txt,len_orig_txt=len_orig_txt,len_summary=len_summary)



#text generate with ai------------
@app.route('/text_generator', methods=['GET', 'POST'])
def text_generator():
    if request.method == 'POST':
        question = request.form['question']
        response_text = generate_content(question)
        return render_template('text_generator.html', question=question, response_text=response_text)
    return render_template('text_generator.html')


#medical help with ai-------------
@app.route('/medicalhelp', methods=['GET', 'POST'])
def medicalhelp():
    if request.method == 'POST':
        try:
            user_input = request.form['user_input']
        except KeyError:
            user_input = ""

        prompt = f"""Imagine you are a medical expert and you are giving accurate medical advice to a patient. 
        You are presented with a medical query and asked to provide a response with a detailed explanation. 
        Note that dont mention any inaccurate or misleading information.

        Medical Query: {user_input}

        Key Details:
        - Provide precise information related to the patient's medical concern.
        - Indicate if any diagnostic tests or examinations have been performed.
        - Specify the current medications or treatments prescribed.
        - The response should be in a paragraph format but not in point-wise.
        - If only a specific disease name is mentioned, response must contain the symptoms, causes, and treatment of the disease with respective headings.

        Guidelines:
        - Use clear and concise language.
        - The vocabulary should be appropriate for the medical context.
        - Include specific parameters or considerations within the medical context.
        - If the response contains a list of items, convert it into a paragraph format.
        - Avoid using abbreviations or acronyms.
        - Avoid Headings and Sub hheadings just give me the complete response in a paragraph format.
        - Refrain from presenting inaccurate or ambiguous information.
        - Ensure the query is focused and not overly broad."""

        gemini_response = get_medical(prompt)
        return render_template('medical.html', user_input=user_input, response=gemini_response)

    return render_template('medical.html')



#image question gen ai-----------------
@app.route('/imagegen', methods=['GET', 'POST'])
def imagegen():
    if request.method == 'POST':
        question = request.form['question']
        file = request.files['image']
        if file:
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            response_text = gen(question, image_path)
            image_url = url_for('static', filename=f'uploads/{filename}')
            return render_template('imagegen.html', question=question, response_text=response_text, image_url=image_url)
    return render_template('imagegen.html')




#pdf question genai--------------
@app.route('/askpdf', methods=['GET', 'POST'])
def askpdf():
    if request.method == 'POST':
        query = request.form['query']
        pdf_file = request.files['pdf_file']
        if pdf_file:
            model_path = "models1/all-MiniLM-L6-v2"  
            response_text = process_pdf(pdf_file, query, model_path)
            return render_template('askpdf.html', query=query, response_text=response_text)
    return render_template('askpdf.html')


if __name__ == '__main__':
    app.run(debug=True)
