from flask import Flask, request, jsonify
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import os

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer dari Hugging Face Hub
MODEL_REPO = "fhru/indobert-judi-online"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

# Fungsi prediksi dengan confidence score
def predict_comments(texts):
    inputs = tokenizer(
        texts,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=128
    )

    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    preds = tf.argmax(probs, axis=1).numpy()
    confidences = tf.reduce_max(probs, axis=1).numpy()

    results = []
    for text, pred, conf in zip(texts, preds, confidences):
        label = "Judi Online" if pred == 1 else "Bukan Judi Online"
        results.append({
            "text": text,
            "label": label,
            "confidence": float(conf)
        })

    return results

# Route home
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "IndoBERT Judi Online Classifier API"}), 200

# Route untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    result = predict_comments(texts)
    return jsonify(result), 200

# Jalankan server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
