import os
import sys
import joblib

PROJECT_ROOT = r"C:\FACULTATE\ANUL 3 SEM 1\RN"
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.pkl')
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, 'models', 'vectorizer.pkl')


def incarca_resurse():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("EROARE: Nu gasesc modelul sau vectorizatorul. Ruleaza train.py!")
        return None, None

    print("Se incarca modelul Neural Network ")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def ruleaza_aplicatie():
    model, vectorizer = incarca_resurse()
    if model is None: return

    print("\n=== SISTEM DE NOTARE AUTOMATA ===")
    print("Scrie 'exit' pentru a iesi.\n")

    while True:
        print("-" * 50)
        barem = input("Raspuns Corect: ").strip()
        if barem.lower() == 'exit': break

        student = input("Raspuns Student: ").strip()
        if student.lower() == 'exit': break

        try:
            text_combinat = [student + " " + barem]
            vector_input = vectorizer.transform(text_combinat)

            nota = model.predict(vector_input)[0]

            nota = max(1.0, min(5.0, nota))

            print(f"\n>>> NOTA CALCULATA: {nota:.2f}")

        except Exception as e:
            print(f"Eroare: {e}")


if __name__ == "__main__":
    ruleaza_aplicatie()
