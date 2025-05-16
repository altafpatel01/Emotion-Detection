import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, '..', 'data', 'goemotions_1.csv')
model_dir = os.path.join(base_dir, '..', 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'emotion_model.pkl')

# Load dataset
df = pd.read_csv(data_path)

emotions= ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                   'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                   'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                   'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                   'relief', 'remorse', 'sadness', 'surprise', 'neutral']

df['emotion'] = df[emotions].idxmax(axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['emotion'], test_size=0.2, random_state=42, stratify=df['emotion']
)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 2)
    )),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)


print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained pipeline
joblib.dump(pipe, model_path)
print(f"Model saved to: {model_path}")
