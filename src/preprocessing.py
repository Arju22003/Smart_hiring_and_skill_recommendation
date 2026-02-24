import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_data(resume_path, job_path):

    resume_data = pd.read_csv(resume_path)
    job_data = pd.read_csv(job_path)

    # Clean resume text
    resume_data['cleaned_resume'] = resume_data['Resume'].apply(clean_text)

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(resume_data['cleaned_resume'])

    # Encode category
    le = LabelEncoder()
    resume_data['encoded_category'] = le.fit_transform(resume_data['Category'])

    # Combine job text
    job_data['job_text'] = (
        job_data['Title'] + " " +
        job_data['Skills'] + " " +
        job_data['Responsibilities'] + " " +
        job_data['Keywords']
    )

    job_data['cleaned_job'] = job_data['job_text'].apply(clean_text)
    job_vectors = tfidf.transform(job_data['cleaned_job'])

    return resume_data, job_data, X, job_vectors, le, tfidf