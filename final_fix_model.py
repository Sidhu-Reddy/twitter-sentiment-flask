import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Load tokenizer (just to get vocab size)
tokenizer = joblib.load("model/tokenizer.pkl")
vocab_size = len(tokenizer.word_index) + 1

# Build clean model from scratch
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=30),
    LSTM(64),
    Dense(3, activation='softmax')
])

# ⚠️ IMPORTANT: DO NOT load old model structure
# Instead, just save fresh model

model.save("model/brand_new_model.keras")

print("✅ BRAND NEW CLEAN MODEL CREATED")