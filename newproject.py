from flask import Flask, jsonify
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import spacy
from flask import request

# Load the models
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading models: {e}")

# Initialize Flask app
app = Flask(__name__)

# Load the data
job_data = pd.read_csv('/app/data/job_descriptions/job_descriptions.csv')  # Update based on your dataset
resumes_data = pd.read_csv('/app/data/resumes/resumes.csv')

@app.route('/')
def home():
    return "Welcome to the ATS App!"

@app.route('/job_descriptions')
def get_job_descriptions():
    return jsonify(job_data.to_dict(orient='records'))

@app.route('/resumes')
def get_resumes():
    return jsonify(resumes_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
