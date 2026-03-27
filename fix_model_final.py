from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import load_model

# Load old model
old_model = load_model("model/sentiment_model.h5", compile=False)

# Rebuild architecture manually (match your model)
new_model = Sequential()

for layer in old_model.layers:
    config = layer.get_config()

    # Remove problematic key
    config.pop("batch_shape", None)

    new_layer = layer.__class__.from_config(config)
    new_model.add(new_layer)

# Build model
new_model.build((None, 30))

# Copy weights
for new_layer, old_layer in zip(new_model.layers, old_model.layers):
    new_layer.set_weights(old_layer.get_weights())

# Save clean model
new_model.save("model/sentiment_model_clean.keras")

print("✅ Clean model saved")