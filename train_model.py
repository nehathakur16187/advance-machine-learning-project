import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load dataset
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
df.columns = ['label', 'message']

# Encode the labels (spam/ham)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Prepare the text data for LSTM
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Tokenize and pad the sequences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=100, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=100, padding='post', truncating='post')

# Build the LSTM model with Bidirectional layer and GRU
model = Sequential([
    Embedding(5000, 64, input_length=100),  # Increased embedding dimension
    Bidirectional(LSTM(64, return_sequences=True)),  # Bidirectional LSTM layer
    Dropout(0.5),
    GRU(64, return_sequences=False),  # Added GRU layer
    Dropout(0.5),
    Dense(64, activation='relu'),  # Increased Dense layer units
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Increased patience
model.fit(X_train_padded, y_train, epochs=15, validation_data=(X_test_padded, y_test), batch_size=64, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model and tokenizer
model.save('spam_lstm_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
