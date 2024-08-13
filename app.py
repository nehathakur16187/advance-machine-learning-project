from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, static_folder='Content')

# Load the trained model and tokenizer
model = tf.keras.models.load_model('Model/spam_lstm_model.h5')
with open('Model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    print(data)
    sequence = tokenizer.texts_to_sequences([data])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    
    prediction = model.predict(padded_sequence)[0][0]
    is_spam = prediction > 0.05
    print(prediction)
    print(is_spam)
    
    return jsonify({'spam': bool(is_spam), 'confidence': float(prediction)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
