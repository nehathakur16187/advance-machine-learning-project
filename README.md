**Spam Detection Model with LSTM**
**Overview**
This project implements a deep learning model to classify SMS messages as either spam or ham (not spam). The model is built using TensorFlow's Keras API, leveraging Bidirectional LSTM layers for effective text classification. The dataset used in this project is the SMS Spam Collection dataset.

**Key Components**
**1. Data Preparation**
**Description:** The dataset is loaded and processed to prepare it for model training. The labels are encoded into binary values, and the text data is tokenized and padded.
            **Libraries:**
            pandas: For data manipulation and analysis.
            numpy: For numerical operations.
            scikit-learn: For splitting the dataset and encoding labels.
            tensorflow: For deep learning model creation and training.
            pickle: For saving the tokenizer used for text preprocessing.
**2. Model Architecture**
**Description:** The model is built using a Sequential API from Keras with the following layers:
                 Embedding Layer: Converts the input text data into dense vectors of fixed size.
                 Bidirectional LSTM Layers: Captures dependencies in both forward and backward directions, making it effective for sequential data processing.
                 Dropout Layers: Helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
                 Dense Layers: Fully connected layers with ReLU activation for intermediate processing and sigmoid activation for binary classification.
**3. Model Training and Evaluation**
                **Training:** The model is trained on 80% of the dataset and validated on the remaining 20%. Early stopping is used to prevent overfitting by monitoring the validation loss.
                **Evaluation:** The model is evaluated on the test dataset, and the accuracy is printed as output.
**4. Model Saving and Loading**
**Description:** After training, the model and the tokenizer are saved for future use. This allows for easy deployment and prediction on new data.
        **Files:**
               **spam_lstm_model.h5:** The saved LSTM model.
               **tokenizer.pkl:** The saved tokenizer for text preprocessing.
**5. Making Predictions**
**Description:** The model can be loaded and used to predict whether a new SMS message is spam or ham. An example is provided where a new message is passed through the model to determine the probability of it being spam.
                Installation
                Prerequisites
                Required Libraries:
                pandas
                numpy
                scikit-learn
                tensorflow
                pickle
