# Dockerfile

# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies and additional tools
RUN apt-get update && apt-get install -y git

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Clone the necessary repositories directly
RUN git clone https://github.com/masnaashraf/Resume-parser.git /app/data/resumes && \
    git clone https://github.com/giterdun345/Job-Description-Skills-Extractor.git /app/data/job_descriptions

# Copy the rest of the application code
COPY . .

# Run the application with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "newproject:app"]
