from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('spam_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['message']
    sequence = tokenizer.texts_to_sequences([data])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence)[0][0]
    is_spam = prediction > 0.5
    
    return jsonify({'spam': bool(is_spam), 'confidence': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
