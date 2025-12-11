import torch
import os
import sys

# Adaugam directorul radacina la path pentru importuri corecte
sys.path.append(os.getcwd())

from src.neural_network.model import ASAGTransformerModel

MODEL_PATH = 'models/trained_model.pth'

def incarca_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispozitiv detectat: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"EROARE: Nu gasesc fisierul {MODEL_PATH}")
        print("Ruleaza intai train.py pentru a genera modelul.")
        return None

    try:
        model = ASAGTransformerModel()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model antrenat incarcat cu succes.")
        return model
    except Exception as e:
        print(f"Eroare la incarcarea modelului: {e}")
        return None

def ruleaza_aplicatie():
    model = incarca_model()
    if model is None:
        return

    print("\n=== SISTEM DE EVALUARE AUTOMATA (ASAG) ===")
    print("Scrie 'exit' pentru a inchide aplicatia.\n")

    while True:
        print("-" * 50)
        
        barem = input("Introdu Raspunsul Corect (Barem): ").strip()
        if barem.lower() == 'exit': break
        if len(barem) < 2:
            print("Text prea scurt.")
            continue

        student = input("Introdu Raspunsul Studentului:    ").strip()
        if student.lower() == 'exit': break
        if len(student) < 2:
            print("Text prea scurt.")
            continue

        try:
            with torch.no_grad():
                # Modelul asteapta liste de string-uri
                score = model([student], [barem])
                
                nota_finala = score.item()
                
                # Limitare vizuala intre 1 si 5
                nota_finala = max(1.0, min(5.0, nota_finala))

            print(f"\n>>> NOTA PREDICTATA: {nota_finala:.2f} / 5.00")
            
            if nota_finala >= 4.5:
                print("Feedback: Excelent!")
            elif nota_finala >= 2.5:
                print("Feedback: Raspuns acceptabil.")
            else:
                print("Feedback: Raspuns incorect sau incomplet.")

        except Exception as e:
            print(f"Eroare la procesare: {e}")

if __name__ == "__main__":
    ruleaza_aplicatie()
