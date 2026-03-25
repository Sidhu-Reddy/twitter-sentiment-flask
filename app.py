from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import re
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
app.secret_key = "twitter_sentiment_secret"

# ================================
# LOAD MODEL & TOKENIZER
# ================================
model = load_model("model/sentiment_model.h5", compile=False)
tokenizer = joblib.load("model/tokenizer.pkl")

# ================================
# LOAD DATASET
# ================================
df = pd.read_csv(
    "dataset/topic_sentiment_data.csv",
    engine="python",
    sep=",",
    quotechar='"',
    on_bad_lines="skip"
)

df.columns = df.columns.str.strip().str.lower()

df["topic"] = (
    df["topic"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.lower()
)

# ================================
# CLEAN TEXT
# ================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

df["cleaned"] = df["tweet"].apply(clean_text)

# ================================
# PREDICTION FUNCTION (UNCHANGED)
# ================================
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=30)
    pred = float(model.predict(padded, verbose=0)[0][0])

    if pred >= 0.65:
        sentiment = "Positive"
    elif pred <= 0.35:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = round(max(pred, 1 - pred) * 100, 1)
    return sentiment, confidence

# ================================
# HOME PAGE
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    summary = None

    if request.method == "POST":
        keyword = re.sub(r"\s+", "", request.form["keyword"].lower())

        data = df[df["topic"] == keyword].copy()

        if data.empty:
            data = df[df["tweet"].str.contains(keyword, case=False, na=False)].copy()

        if data.empty:
            data = df.sample(min(10, len(df)))

        # ================================
        # 🔥 FAST + SAFE BATCH PREDICTION
        # ================================
        texts = list(data["cleaned"])

        seq = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seq, maxlen=30)

        preds = model.predict(padded, verbose=0)

        # ✅ FIX SHAPE ISSUE (IMPORTANT)
        if len(preds.shape) > 1:
            preds = preds[:, 0]

        preds = preds[:len(data)]  # ensure exact match

        sentiments = []
        confidences = []

        for pred in preds:
            if pred >= 0.65:
                sentiment = "Positive"
            elif pred <= 0.35:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            confidence = round(max(pred, 1 - pred) * 100, 1)

            sentiments.append(sentiment)
            confidences.append(confidence)

        data["sentiment"] = sentiments
        data["confidence"] = confidences

        counts = data["sentiment"].value_counts()
        total = len(data)

        summary = {
            "keyword": keyword,
            "Positive": int(counts.get("Positive", 0)),
            "Neutral": int(counts.get("Neutral", 0)),
            "Negative": int(counts.get("Negative", 0)),
            "Positive_pct": int((counts.get("Positive", 0) / total) * 100),
            "Neutral_pct": int((counts.get("Neutral", 0) / total) * 100),
            "Negative_pct": int((counts.get("Negative", 0) / total) * 100),
            "Confidence": round(np.mean(confidences), 1)
        }

        session["results"] = [
            {
                "text": str(row["tweet"]),
                "sentiment": str(row["sentiment"]),
                "confidence": float(row["confidence"])
            }
            for _, row in data.iterrows()
        ]

        session["keyword"] = keyword

    return render_template("index.html", summary=summary)

# ================================
# TWEETS PAGE
# ================================
@app.route("/tweets")
def tweets():
    results = session.get("results", [])
    keyword = session.get("keyword", "")

    if not results:
        return redirect(url_for("index"))

    return render_template(
        "tweets.html",
        tweets=results,
        keyword=keyword
    )

# ================================
# CLEAR SESSION
# ================================
@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for("index"))

# ================================
# RUN
# ================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)