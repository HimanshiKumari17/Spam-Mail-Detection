# spam_detection.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


# Clean text function
def clean_text(message):
    message = message.lower()
    message = ''.join([char for char in message if char not in string.punctuation])
    words = message.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Function to predict spam
def predict_spam(email, vectorizer, model):
    email_clean = clean_text(email)
    email_vector = vectorizer.transform([email_clean])
    prediction = model.predict(email_vector)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'


# Main function
def main():
    # Download stopwords (only once)
    nltk.download('stopwords')

    # Load dataset
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    # Clean the text
    data['clean_message'] = data['message'].apply(clean_text)

    # Convert text to numbers
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['clean_message'])

    # Prepare labels
    data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})
    y = data['label_num']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Get metrics for plotting
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Plot the bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8,5))
    plt.bar(metrics, scores, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Scores are between 0 and 1

    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.show()

    # Test your own emails
    print("\n--- Testing custom emails ---")
    test_email_1 = "Congratulations! You've won a free iPhone. Click here to claim now."
    test_email_2 = "Hi friend, let's catch up tomorrow over coffee."

    print(f"Email: {test_email_1}\nPrediction: {predict_spam(test_email_1, vectorizer, model)}\n")
    print(f"Email: {test_email_2}\nPrediction: {predict_spam(test_email_2, vectorizer, model)}\n")


# Run the main function
if __name__ == "__main__":
    main()
