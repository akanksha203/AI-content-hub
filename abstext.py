from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarizer2(rawtext):
    # Load PEGASUS model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained('models/pegasus_tokenizer')
        model = AutoModelForSeq2SeqLM.from_pretrained('models/pegasus_model')

    except:
        model_name = 'google/pegasus-cnn_dailymail'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Save the model and tokenizer
        
        tokenizer.save_pretrained('models/pegasus_tokenizer')
        model.save_pretrained('models/pegasus_model')

    
    # Tokenize and encode data
    inputs = tokenizer.encode(rawtext, return_tensors='pt', max_length=1024, truncation=True)

    
    summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    

    return summary,rawtext,len(rawtext.split()),len(summary.split())
















# from transformers import pipeline

# def summarizer2(rawtext):
#     summarizer=pipeline("summarization",model='t5-base',tokenizer='t5-base',framework='pt')

#     summary=summarizer(rawtext,max_length=500,min_length=30,do_sample=False)

#     return summary[0]['summary_text'],rawtext, len(rawtext.split(' ')),len(summary[0]['summary_text'].split(' '))



