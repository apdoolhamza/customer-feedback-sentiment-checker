from flask import Flask, jsonify, render_template, request
import joblib
import string
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(__file__)
nltk.data.path.append(os.path.join(BASE_DIR, "nltk_data"))
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }

model = joblib.load(os.path.join(BASE_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

app = Flask(__name__)

TEXT_COLUMN_ALIASES = (
    "feedback",
    "feedback_text",
    "review",
    "comment",
    "message",
    "text",
    "Tweet Text",
)


def preprocess(text):
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def find_text_column(columns):
    normalized_columns = {str(column).strip().casefold(): column for column in columns}
    for alias in TEXT_COLUMN_ALIASES:
        matched_column = normalized_columns.get(alias.casefold())
        if matched_column is not None:
            return matched_column
    accepted_columns = ", ".join(TEXT_COLUMN_ALIASES)
    raise ValueError(f"CSV must include one of these text columns: {accepted_columns}")


def predict_feedback(text):
    cleaned = preprocess(text)
    if not cleaned:
        raise ValueError("Feedback text is required.")
    vector = vectorizer.transform([cleaned])
    proba = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    class_index = list(model.classes_).index(pred)
    confidence = round(float(proba[class_index]) * 100, 2)
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "cleaned_feedback": cleaned,
    }


def predict_dataframe(df):
    text_column = find_text_column(df.columns)
    clean_df = df.copy()
    clean_df["feedback"] = clean_df[text_column].fillna("").astype(str).str.strip()
    clean_df = clean_df[clean_df["feedback"] != ""]
    clean_df = clean_df.drop_duplicates(subset=["feedback"])
    results = []
    for feedback in clean_df["feedback"].tolist():
        prediction = predict_feedback(feedback)
        prediction["feedback"] = feedback
        results.append(prediction)
    return results


def load_batch_dataframe():
    if "file" in request.files:
        try:
            return pd.read_csv(request.files["file"])
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as exc:
            raise ValueError("Upload a valid UTF-8 CSV file.") from exc

    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSON rows must be an array of objects.")
    return pd.DataFrame(rows)


@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    confidence = None

    if request.method == "POST":
        text = request.form["feedback"]
        try:
            prediction = predict_feedback(text)
            sentiment = prediction["sentiment"]
            confidence = prediction["confidence"]
        except ValueError as exc:
            sentiment = str(exc)

    return render_template("index.html", sentiment=sentiment, confidence=confidence)


@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    data = request.get_json(silent=True) or {}
    text = data.get("feedback") or data.get("text") or data.get("Tweet Text")
    try:
        return jsonify(predict_feedback(text))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/sentiment/batch", methods=["POST"])
def api_sentiment_batch():
    try:
        results = predict_dataframe(load_batch_dataframe())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
