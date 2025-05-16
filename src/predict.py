import os
import joblib

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'model', 'emotion_model.pkl')

# Load the trained model
model = joblib.load(model_path)

while True:
    text = input("Enter text (or 'exit' to quit): ").strip()
    if text.lower() == 'exit':
        break
    pred = model.predict([text])[0]
    print(f"Predicted Emotion: {pred}")
