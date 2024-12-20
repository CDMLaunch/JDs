#   Overview


![Resume Parser](https://github.com/masnaashraf/Resume-parser/blob/main/resume-parsing-768x310.jpeg)


The Resume Matcher is a Python-based project designed to match job descriptions with candidate resumes using natural language processing techniques. This tool aids in identifying the most suitable candidates for specific job roles by calculating the similarity between job descriptions and candidate resumes.

The project is divided into four main steps:

##  Data Collection:

-   Extract resume keywords using PyPDF2 and SpaCy libraries.
-   Fetch job description data from HuggingFace dataset library.

##  Text Preprocessing and Tokenization:

-Preprocess and tokenize both resumes and job descriptions.
-Ensure consistent text formatting and language handling.

##  Word Embedding Extraction:

-Generate word embeddings for both resumes and job descriptions.
-Utilize advanced models like DistilBERT for embeddings.

##  Resume Matching:

-Calculate cosine similarity between job descriptions and resumes.
-Rank CVs based on similarity scores and list the top candidates.


##  Project Structure

The project directory structure is organized as follows:

-   Data_fetching/: Contains code for extracting resume data and job description data using PyPDF2 and the HuggingFace dataset library.
-   Data_preprocessing: Contains code for text preprocessing and tokenization of resumes and job descriptions.
-   word_embeddings: Contains code for generating word embeddings from preprocessed text data.
-   job_matcher: Contains the main script and modules for matching resumes with job descriptions.
-   You can navigate to these directories to access the specific code related to different project stages. Each folder contains code files and modules to perform its respective task.

# Results

The matching results are stored in the [output/ directory](https://github.com/masnaashraf/Resume-parser/blob/main/candidate_job_matching_results.json) in JSON format. You can explore the top candidates for each job description based on similarity scores.
