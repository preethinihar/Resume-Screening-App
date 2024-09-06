import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')  # Make sure to replace 'your_dataset.csv' with the actual filename

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the text data
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Save the trained model and vectorizer
with open('clf.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('tfidf.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model and vectorizer saved successfully.")
