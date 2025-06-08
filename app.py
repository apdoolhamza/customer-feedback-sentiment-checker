from flask import Flask, render_template, request
import joblib
import string
import nltk
import os

# Load stopwords from nltk_data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Main route: shows form and handles prediction
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    confidence = None

    if request.method == "POST":
        text = request.form["feedback"]
        cleaned = preprocess(text)
        vector = vectorizer.transform([cleaned])
        proba = model.predict_proba(vector)[0]
        pred = model.predict(vector)[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = round(proba[pred] * 100, 2)

    return render_template("index.html", sentiment=sentiment, confidence=confidence)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
