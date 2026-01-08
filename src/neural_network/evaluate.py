import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, ConfusionMatrixDisplay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
METRICS_SAVE_PATH = os.path.join(BASE_DIR, 'results', 'test_metrics.json')
CONF_MATRIX_PATH = os.path.join(BASE_DIR, 'docs', 'confusion_matrix.png')

os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'docs'), exist_ok=True)


def evaluate_model():
    print("--- EVALUARE MODEL & ANALIZA 5 ERORI ---")

    if not os.path.exists(MODEL_PATH):
        print(f" EROARE: Nu gasesc modelul la {MODEL_PATH}")
        return

    print("1. Incarc modelul si datele...")
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        df_test = pd.read_csv(TEST_PATH)
    except Exception as e:
        print(f" Eroare la incarcare: {e}")
        return

    X_text = df_test['answer_student'].astype(str) + " " + df_test['answer_correct'].astype(str)
    y_true = df_test['score_manual'].astype(float)

    X_sparse = vectorizer.transform(X_text)

    X_dense = X_sparse.toarray()
    X_final = np.array(X_dense, dtype=np.float32)

    print("2. Calculez predictiile...")
    y_pred_raw = model.predict(X_final)

    possible_grades = np.array([0.0, 2.5, 4.0, 5.0])

    def get_nearest_grade(prediction):
        idx = (np.abs(possible_grades - prediction)).argmin()
        return possible_grades[idx]

    y_pred_class = np.array([get_nearest_grade(p) for p in y_pred_raw])
    y_true_class = np.array([get_nearest_grade(t) for t in y_true])

    acc = np.mean(y_pred_class == y_true_class)
    f1 = f1_score(y_true_class.astype(str), y_pred_class.astype(str), average='weighted')
    mse = mean_squared_error(y_true, y_pred_raw)  # MSE pe valorile brute

    print(f"\nREZULTATE FINALE:")
    print(f"   Acuratete: {acc * 100:.2f}%")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   MSE Loss:  {mse:.4f}")

    cm = confusion_matrix(y_true_class.astype(str), y_pred_class.astype(str), labels=["0.0", "2.5", "4.0", "5.0"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0.0", "2.5", "4.0", "5.0"])

    plt.figure(figsize=(8, 8))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Matricea de Confuzie')
    plt.savefig(CONF_MATRIX_PATH)
    print(f" Confusion Matrix salvata in: {CONF_MATRIX_PATH}")

    metrics = {
        "test_accuracy": round(acc, 4),
        "test_f1_score": round(f1, 4),
        "test_mse": round(mse, 4)
    }
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 60)
    print("TOP 5 CELE MAI MARI GRESELI (ANALIZA PENTRU BONUS)")
    print("=" * 60)

    df_test['pred_class'] = y_pred_class
    df_test['diff'] = np.abs(df_test['score_manual'] - df_test['pred_class'])

    worst_mistakes = df_test.sort_values(by='diff', ascending=False).head(5)

    for i, row in worst_mistakes.iterrows():
        print(f"\n[Exemplu #{i}]")
        print(f"Intrebare ID: {row.get('question_id', 'N/A')}")
        print(f"Raspuns Student: '{row['answer_student']}'")
        print(f"Nota REALA: {row['score_manual']}  |  Nota AI: {row['pred_class']}")
        print(f"Diferenta: {row['diff']}")
        print("-" * 30)


if __name__ == "__main__":
    evaluate_model()
