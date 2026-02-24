from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import preprocess_data

# Load & preprocess
resume_path = "../dataset/raw/UpdatedResumeDataSet.csv"
job_path = "../dataset/raw/job_dataset.csv"

resume_data, job_data, X, job_vectors, le, tfidf = preprocess_data(
    resume_path,
    job_path
)

y = resume_data['encoded_category']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)