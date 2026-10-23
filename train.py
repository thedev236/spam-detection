import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def train_and_save_model(data_path="dataset/spam.csv"):
    # Load dataset (expects columns 'label' and 'text')
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Standardize column names if using typical Kaggle datasets
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

    # Preprocessing: Map labels
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_num'], test_size=0.2, random_state=42
    )

    # Feature extraction
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model training
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluation
    predictions = model.predict(X_test_vec)
    print("--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))

    # Save artifacts
    joblib.dump(model, "model/classifier.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("Model and Vectorizer saved to model/ directory.")

if __name__ == "__main__":
    train_and_save_model()