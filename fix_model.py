from tensorflow.keras.models import load_model

# Load old model
model = load_model("model/sentiment_model.h5", compile=False)

# Save in new format (IMPORTANT)
model.save("model/sentiment_model.keras")