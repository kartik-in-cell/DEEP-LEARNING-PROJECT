# DEEP-LEARNING-PROJECT

Project 2: NLP Text Classification (Sentiment Analysis)

Overview

This project implements a Deep Learning-based Text Classification Model using TensorFlow and Keras. It classifies IMDb movie reviews into positive or negative sentiments using an LSTM-based neural network.

Objectives

Load and preprocess textual data.

Train an LSTM-based model for binary sentiment classification.

Evaluate model performance.

Visualize accuracy and loss trends during training.

Technologies Used

Python

TensorFlow/Keras

Matplotlib

Dataset Used

IMDb Movie Reviews Dataset (50,000 reviews: 25,000 for training, 25,000 for testing).

Reviews are converted into sequences using Tokenization & Padding.

Implementation Steps

Load Data: Use imdb.load_data() to fetch movie reviews.

Preprocessing:

Tokenization & padding sequences to equal length.

Convert words into numerical indices.

Model Building:

Use an Embedding Layer to represent words.

Add LSTM layers for sequential processing.

Use a Dense layer with sigmoid activation for binary classification.

Training & Evaluation:

Train the model using binary_crossentropy loss.

Plot accuracy trends.

Save the trained model for future use.

Files Included

nlp_text_classification.py: Python script for training the NLP model.

sentiment_analysis_model.h5: Trained LSTM model.

How to Run

Run nlp_text_classification.py using Python.

The model will be trained and saved as sentiment_analysis_model.h5.

Training results will be visualized using Matplotlib.

Future Improvements

Use a custom dataset for text classification.

Implement a more advanced Transformer-based model (BERT, GPT).

Deploy the model as an API for real-world applications.

Conclusion

These two projects demonstrate essential Data Science and Deep Learning skills:

The Data Pipeline ensures structured data preprocessing.

The NLP Model automates sentiment analysis using deep learning.

By implementing these, you gain practical experience in ETL processes, text preprocessing, and deep learning model development. 🚀

