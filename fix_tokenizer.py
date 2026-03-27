import joblib
from tensorflow.keras.preprocessing.text import Tokenizer

# load old tokenizer
old_tokenizer = joblib.load("model/tokenizer.pkl")

# recreate new tokenizer
new_tokenizer = Tokenizer()
new_tokenizer.word_index = old_tokenizer.word_index

# save again
joblib.dump(new_tokenizer, "model/tokenizer_fixed.pkl")

print("Tokenizer fixed and saved!")