# Dockerfile

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Download BERT tokenizer and model
RUN python -c "from transformers import BertTokenizer, BertModel; BertTokenizer.from_pretrained('bert-base-multilingual-cased'); BertModel.from_pretrained('bert-base-multilingual-cased')"

# Clone the repository directly from GitHub
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/masnaashraf/Resume-parser.git && \
    git clone https://github.com/giterdun345/Job-Description-Skills-Extractor.git

# Copy the rest of the application code
COPY . .

# Run the application with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "newproject:app"]
