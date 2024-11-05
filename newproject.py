from flask import Flask, jsonify, render_template_string
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from flask import request
import os

# Load models for advanced feature extraction
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    print(f"Error loading models: {e}")

app = Flask(__name__)

# Data loading and processing
job_data_path = 'data/collected_data/indeed_job_dataset.csv'
resumes_data_path = 'Resume-Corpus-Dataset/data-files/predictions1.csv'

try:
    job_reqs = pd.read_csv(job_data_path)['job_description'].tolist()
    resumes = pd.read_csv(resumes_data_path)['resume_text'].tolist()
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    job_reqs = [
        "Data analyst with experience in SQL, Power BI, Python, and data visualization techniques.",
        "Software engineer needed with expertise in Java, Spring Boot, REST APIs, and AWS cloud.",
        "HR specialist with a focus on employee relations and compliance.",
        "Marketing specialist with experience in social media management, content creation, SEO, and Google Analytics.",
        "Sales representative with a proven track record in B2B sales and strong CRM experience."
    ]
    resumes = [
        "Experienced data analyst skilled in SQL, Power BI, Python, and data visualization.",
        "Software engineer with 5 years of experience in Java, Spring Boot, REST APIs, and AWS cloud.",
        "HR specialist with 7 years of experience in employee relations and compliance.",
        "Marketing specialist experienced in social media management, SEO, and Google Analytics.",
        "B2B sales representative with 10 years of experience in sales and account management."
    ]

@app.route('/')
def home():
    response = """
    <h1>Welcome to the AI ATS Dashboard</h1>
    <p>Use the following endpoints to view data:</p>
    <ul>
        <li><a href='/rankings'>View Resume and Job Rankings</a></li>
        <li><a href='/dispositions'>View Dispositions</a></li>
    </ul>
    """
    return response

@app.route('/rankings')
def display_rankings():
    rankings = rank_resumes_to_jobs(resumes, job_reqs)
    # Generate a bar plot for each resume ranking
    plot_urls = []
    for idx, ranking in enumerate(rankings):
        plt.figure(figsize=(10, 5))
        sns.barplot(x=[f"Job {i + 1}" for i in range(len(ranking))], y=ranking, palette='viridis')
        plt.title(f'Resume {idx + 1} Ranking for Job Descriptions')
        plt.xlabel('Job Descriptions')
        plt.ylabel('Ranking Score')
        # Save plot to a string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plot_urls.append(plot_url)
        plt.close()

    # Generate HTML with embedded images
    response = """
    <h2>Resume Rankings</h2>
    <div>
    """
    for idx, plot_url in enumerate(plot_urls):
        response += f"<h3>Resume {idx + 1}</h3><img src='data:image/png;base64,{plot_url}'/><br><br>"
    response += "</div>"
    return render_template_string(response)

@app.route('/dispositions')
def display_dispositions():
    # Create a summary of dispositions
    summary = {job: 0 for job in job_reqs}
    for resume in resumes:
        for job in job_reqs:
            summary[job] += 1

    # Generate a bar plot for dispositions
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(summary.keys()), y=list(summary.values()), palette='viridis')
    plt.title('Disposition Summary for Job Descriptions')
    plt.xlabel('Job Descriptions')
    plt.ylabel('Number of Resumes Reviewed')
    plt.xticks(rotation=45, ha='right')
    # Save plot to a string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Generate HTML with embedded image
    response = f"""
    <h2>Disposition Summary</h2>
    <div><img src='data:image/png;base64,{plot_url}'/><br><br></div>
    <ul>
    """
    for resume in resumes:
        response += f"<li><strong>Resume:</strong> {resume}</li><ul>"
        for job in job_reqs:
            response += f"<li><strong>Job Description:</strong> {job} - <strong>Disposition:</strong> Keep for Review</li>"
        response += "</ul>"
    response += "</ul>"
    return render_template_string(response)

def rank_resumes_to_jobs(resumes, job_reqs):
    rankings = []
    for resume in resumes:
        scores = []
        for job in job_reqs:
            resume_emb = extract_features(resume)
            job_emb = extract_features(job)
            score = cosine_similarity(resume_emb, job_emb)
            scores.append(score)
        ranking = np.argsort(scores)[::-1]
        rankings.append([scores[i] for i in ranking])
    return rankings

def extract_features(text):
    tokens = tokenizer(text, return_tensors='pt')
    outputs = bert_model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
