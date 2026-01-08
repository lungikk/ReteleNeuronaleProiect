import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = r"C:\FACULTATE\ANUL 3 SEM 1\RN"

TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'train', 'train.csv')
VAL_PATH = os.path.join(PROJECT_ROOT, 'data', 'validation', 'validation.csv')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.pkl')
VECTORIZER_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'vectorizer.pkl')
GRAPH_SAVE_PATH = os.path.join(PROJECT_ROOT, 'docs', 'loss_curve.png')

os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'docs'), exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nu gasesc fisierul: {path}")
    df = pd.read_csv(path)
    X_text = df['answer_student'].astype(str) + " " + df['answer_correct'].astype(str)
    y = df['score_manual'].astype(float)
    return X_text, y


if __name__ == "__main__":
    print("--- ANTRENARE NIVEL 2 (AUGMENTARE + ARRAY FIX) ---")

    print("1. Incarc datele...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_val_text, y_val = load_data(VAL_PATH)

    print("2. Procesez textul (Vectorizare)...")
    vectorizer = TfidfVectorizer(max_features=1000)

    X_train_sparse = vectorizer.fit_transform(X_train_text)

    print("   -> [Nivel 2] Aplic augmentare: Zgomot Gaussian...")

    X_train_dense = X_train_sparse.toarray()

    noise = np.random.normal(0, 0.005, X_train_dense.shape)

    X_train_aug = X_train_dense + noise

    X_train_final = np.array(X_train_aug, dtype=np.float32)
    y_train_final = np.array(y_train, dtype=np.float32)

    print(f"   -> Verificare tip date: {type(X_train_final)}")

    print("3. Configurez Reteaua Neuronala...")
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='sgd',
        learning_rate='adaptive',
        learning_rate_init=0.01,
        momentum=0.9,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.15,
        max_iter=300,
        batch_size=32,
        random_state=42,
        verbose=True
    )

    print("4. Start Antrenare...")
    start_time = time.time()

    model.fit(X_train_final, y_train_final)

    duration = time.time() - start_time
    print(f"\n GATA! Antrenare finalizata in {duration:.2f} secunde!")
    print(f"   Epoci rulate: {model.n_iter_}")

    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)

    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.title('Curba de Invatare (Nivel 2 - Augmentat)')
        plt.xlabel('Epoci')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(GRAPH_SAVE_PATH)
        print(f" Grafic salvat: {GRAPH_SAVE_PATH}")
