import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset/topic_sentiment_data.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.lower()

texts = df["tweet"].astype(str)
labels = df["sentiment"]

# =========================
# ENCODE LABELS (3 CLASS)
# =========================
le = LabelEncoder()
y = le.fit_transform(labels)  # 0,1,2
y = to_categorical(y, num_classes=3)

# =========================
# TOKENIZER
# =========================
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=100)

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODEL (3 OUTPUT)
# =========================
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 🔥 IMPORTANT
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# =========================
# TRAIN
# =========================
model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =========================
# SAVE
# =========================
model.save("model/twitter_sentiment_model.h5")

with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model trained and saved!")