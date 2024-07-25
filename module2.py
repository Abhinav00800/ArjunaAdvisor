# from flask import Flask, request, jsonify, render_template
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from module3 import extract_text_from_pdf, clean_text, preprocess_text
# import os

# app = Flask(__name__)

# # Load the model and tokenizer
# model_path = './saved_model'
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model directory '{model_path}' not found.")

# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     pdf_path = data.get('pdf_path')
#     query = data.get('query')

#     if not pdf_path or not query:
#         return jsonify({'error': 'pdf_path and query are required'}), 400

#     try:
#         raw_text = extract_text_from_pdf(pdf_path)
#         cleaned_text = clean_text(raw_text)
#         processed_text = preprocess_text(cleaned_text)

#         inputs = tokenizer(processed_text, return_tensors='pt', padding=True, truncation=True)
#         outputs = model(**inputs)
#         logits = outputs.logits
#         prediction = torch.argmax(logits, dim=1).item()

#         response = "This is relevant to the research paper." if prediction == 1 else "This is not relevant to the research paper."
#         return jsonify({'response': response})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from module3 import preprocess_text, answer_question

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        answer = answer_question(question, "preprocess_text")
        return jsonify({'response': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

