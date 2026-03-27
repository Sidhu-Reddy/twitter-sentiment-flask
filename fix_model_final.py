from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model

# Load old model
old_model = load_model("model/sentiment_model.h5", compile=False)

# Build correct architecture (FIXED)
new_model = Sequential([
    Embedding(input_dim=5000, output_dim=64),
    LSTM(64),
    Dense(3, activation='softmax')   # 🔥 FIXED
])

# Build model
new_model.build(input_shape=(None, 30))

# Copy weights
new_model.set_weights(old_model.get_weights())

# Save clean model
new_model.save("model/sentiment_model_final.keras")

print("✅ FINAL CLEAN MODEL CREATED")