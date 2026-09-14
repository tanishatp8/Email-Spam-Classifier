# SMS Spam Classifier

A simple machine learning project that classifies SMS/text messages as **spam** or **ham** (legitimate) using TF-IDF feature extraction and a Multinomial Naive Bayes classifier.

## Overview

This project demonstrates a complete, minimal text-classification pipeline:

1. Load a labelled dataset of SMS messages (`data/sms_data.csv`)
2. Convert text to numerical features using **TF-IDF** (Term Frequency–Inverse Document Frequency)
3. Train a **Multinomial Naive Bayes** classifier — a standard, fast, and effective baseline for text classification
4. Evaluate the model (accuracy, precision, recall, F1-score, confusion matrix)
5. Use the trained model to classify new, unseen messages

## Project Structure

```
spam-classifier/
├── data/
│   └── sms_data.csv       # Labelled dataset (ham/spam)
├── train.py                # Trains the model and prints evaluation metrics
├── predict.py               # Classifies a new message using the saved model
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train the model:**

```bash
python train.py
```

This trains the classifier on `data/sms_data.csv`, prints evaluation metrics on a held-out test split, and saves the trained model as `spam_model.joblib` and `vectorizer.joblib`.

**2. Classify a new message:**

```bash
python predict.py "Congratulations! You have won a free prize, click here to claim now"
```

Example output:
```
Message : "Congratulations! You have won a free prize, click here to claim now"
Prediction: SPAM  (confidence: 78.4%)
```

## How It Works

- **TF-IDF Vectorization**: Converts each message into a numerical vector where words that are frequent in a message but rare across the whole dataset get higher weight — this helps the model pick up on distinctive "spammy" words (e.g. "free", "winner", "click", "urgent").
- **Multinomial Naive Bayes**: A probabilistic classifier well-suited to word-count/frequency style features, commonly used as a strong, lightweight baseline for spam detection and text classification tasks.

## Possible Improvements

- Train on a larger, real-world dataset (e.g. the UCI SMS Spam Collection) for better generalisation
- Add n-gram features (bigrams/trigrams) to the TF-IDF vectorizer
- Try alternative models (Logistic Regression, SVM) and compare performance
- Build a simple web interface (Flask/Streamlit) around `predict.py`

## Author

Tanisha Priya — B.Tech CSE, IILM University, Greater Noida
