import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'train', 'train.csv')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'test', 'test.csv')
MLP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'models', 'vectorizer.pkl')


def get_nearest_grade(prediction):
    possible_grades = np.array([0.0, 2.5, 4.0, 5.0])
    idx = (np.abs(possible_grades - prediction)).argmin()
    return possible_grades[idx]


def evaluate_architecture(name, y_true, y_pred, duration):
    y_pred_fixed = np.array([get_nearest_grade(p) for p in y_pred])
    y_true_fixed = np.array([get_nearest_grade(t) for t in y_true])

    acc = np.mean(y_pred_fixed == y_true_fixed)
    f1 = f1_score(y_true_fixed.astype(str), y_pred_fixed.astype(str), average='weighted')
    mse = mean_squared_error(y_true, y_pred)

    return {
        "Model": name,
        "Acuratete": f"{acc * 100:.2f}%",
        "F1-Score": f"{f1:.4f}",
        "MSE (Eroare)": f"{mse:.4f}",
        "Timp Predictie (s)": f"{duration:.4f}"
    }


if __name__ == "__main__":
    print("--- COMPARARE ARHITECTURI (BONUS) ---")

    print("1. Incarc datele...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    X_train_text = df_train['answer_student'].astype(str) + " " + df_train['answer_correct'].astype(str)
    X_test_text = df_test['answer_student'].astype(str) + " " + df_test['answer_correct'].astype(str)

    y_train = df_train['score_manual'].astype(float)
    y_true = df_test['score_manual'].astype(float)

    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = []

    print("\n2. Evaluez Model A: MLP Regressor (Reteaua Ta)...")
    mlp_model = joblib.load(MLP_MODEL_PATH)

    X_test_dense = X_test.toarray()
    X_test_final = np.array(X_test_dense, dtype=np.float32)

    t0 = time.time()
    y_pred_mlp = mlp_model.predict(X_test_final)
    t1 = time.time()
    results.append(evaluate_architecture("MLP (Neural Network)", y_true, y_pred_mlp, t1 - t0))

    print("3. Antrenez si Evaluez Model B: Random Forest (Baseline)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    rf_model.fit(X_train, y_train)

    t0 = time.time()
    y_pred_rf = rf_model.predict(X_test)
    t1 = time.time()
    results.append(evaluate_architecture("Random Forest", y_true, y_pred_rf, t1 - t0))

    print("\n" + "=" * 85)
    print(f"{'Model':<25} | {'Acuratete':<15} | {'F1-Score':<10} | {'MSE':<10} | {'Timp (s)':<10}")
    print("-" * 85)

    for res in results:
        print(
            f"{res['Model']:<25} | {res['Acuratete']:<15} | {res['F1-Score']:<10} | {res['MSE (Eroare)']:<10} | {res['Timp Predictie (s)']:<10}")
    print("=" * 85)
