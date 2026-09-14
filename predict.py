"""
Classify a new SMS/message as spam or ham using the trained model.

Usage:
    python predict.py "Congratulations! You won a free prize, click here"
"""

import sys
import joblib

MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


def main():
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your message here"')
        sys.exit(1)

    message = sys.argv[1]

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "SPAM" if pred == 1 else "HAM"
    confidence = prob[pred] * 100

    print(f'Message : "{message}"')
    print(f"Prediction: {label}  (confidence: {confidence:.1f}%)")


if __name__ == "__main__":
    main()
