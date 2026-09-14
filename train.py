"""
Train an SMS Spam Classifier using TF-IDF + Multinomial Naive Bayes.

Usage:
    python train.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

DATA_PATH = os.path.join("data", "sms_data.csv")
MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} messages ({(df.label == 'spam').sum()} spam, "
          f"{(df.label == 'ham').sum()} ham)")

    X = df["message"]
    y = df["label"].map({"ham": 0, "spam": 1})

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Vectorize text (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_vec)
    print("\n--- Evaluation on test set ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.2f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.2f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, y_pred))

    # 6. Save model + vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\nSaved model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
