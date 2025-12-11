import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import os
import json

# --- 1. CONFIGURĂRI ---
TEST_PATH = 'data/test/test.csv'
MODEL_PATH = 'models/trained_model.pth'
HISTORY_PATH = 'results/training_history.csv'
METRICS_SAVE_PATH = 'results/test_metrics.json'
GRAPH_SAVE_PATH = 'docs/loss_curve.png'

# Detectare device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. DEFINIREA CLASEI MODELULUI ---
# (Trebuie să fie IDENTICĂ cu cea din train.py pentru a putea încărca greutățile)
class ASAGTransformerModel(nn.Module):
    def __init__(self):
        super(ASAGTransformerModel, self).__init__()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, student_texts, correct_texts):
        with torch.no_grad():
            u = self.encoder.encode(student_texts, convert_to_tensor=True, device=device)
            v = self.encoder.encode(correct_texts, convert_to_tensor=True, device=device)
        combined = torch.cat((u, v), dim=1)
        output = self.regressor(combined)
        return output.squeeze()

# --- 3. FUNCȚIE GENERARE GRAFIC LOSS (Nivel 2 - Obligatoriu) ---
def plot_learning_curve():
    if not os.path.exists(HISTORY_PATH):
        print("⚠️ Nu găsesc istoricul antrenării. Rulați train.py întâi.")
        return

    df = pd.read_csv(HISTORY_PATH)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    
    plt.title('Curba de Învățare (Loss Curve)')
    plt.xlabel('Epoci')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Salvare în folderul docs
    os.makedirs('docs', exist_ok=True)
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"📈 Graficul Loss a fost salvat în: {GRAPH_SAVE_PATH}")
    plt.close()

# --- 4. FUNCȚIE EVALUARE PE TEST SET ---
def evaluate_model():
    print(f"--- Începe Evaluarea pe Test Set ---")
    
    # Încărcare date test
    if not os.path.exists(TEST_PATH):
        print(f"EROARE: Nu găsesc {TEST_PATH}")
        return

    df_test = pd.read_csv(TEST_PATH)
    print(f"Date test încărcate: {len(df_test)} înregistrări")

    # Încărcare Model Antrenat
    model = ASAGTransformerModel().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model antrenat încărcat cu succes.")
    else:
        print("EROARE: Nu găsesc modelul antrenat. Rulați train.py.")
        return

    model.eval()
    
    predictions = []
    actuals = []

    # Inferență (Batch by batch ar fi ideal, dar aici facem totul odată pt simplitate la 200 date)
    # Dacă ai eroare de memorie, împarte în batch-uri ca la train.py
    batch_size = 32
    total_samples = len(df_test)
    
    print("Se rulează predicțiile...")
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch = df_test.iloc[i:i+batch_size]
            
            stud_texts = batch['answer_student'].astype(str).tolist()
            corr_texts = batch['answer_correct'].astype(str).tolist()
            scores = batch['score_manual'].astype(float).tolist()
            
            preds = model(stud_texts, corr_texts)
            
            # Gestionare caz batch=1 sau output scalar
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
                
            predictions.extend(preds.cpu().numpy())
            actuals.extend(scores)

    # --- 5. CALCUL METRICI ---
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 1. MSE (Mean Squared Error)
    mse = mean_squared_error(actuals, predictions)
    
    # 2. MAE (Mean Absolute Error)
    mae = mean_absolute_error(actuals, predictions)
    
    # 3. Pearson Correlation (Cât de bine urmărește trendul notelor)
    corr, _ = pearsonr(actuals, predictions)
    
    # 4. Acuratețe Personalizată (Predicție în marja de 0.5 puncte)
    # Considerăm corect dacă |pred - actual| <= 0.5
    accurate_predictions = np.sum(np.abs(predictions - actuals) <= 0.5)
    accuracy = accurate_predictions / total_samples

    metrics = {
        "test_mse": round(mse, 4),
        "test_mae": round(mae, 4),
        "pearson_correlation": round(corr, 4),
        "test_accuracy_tolerance_0.5": round(accuracy, 4) # Asta e metrica pt Nivel 1
    }

    # Afișare rezultate
    print("\n📊 REZULTATE FINALE PE TEST:")
    print(f"   MSE (Eroare pătratică): {mse:.4f}")
    print(f"   Pearson Correlation:    {corr:.4f} (Țintă > 0.8)")
    print(f"   Acuratețe (marjă 0.5p): {accuracy*100:.2f}% (Țintă >= 65%)")

    # Salvare metrici JSON
    with open(METRICS_SAVE_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metricile au fost salvate în: {METRICS_SAVE_PATH}")

# --- MAIN ---
if __name__ == "__main__":
    # 1. Generăm graficul din istoric
    plot_learning_curve()
    
    # 2. Calculăm notele pe test
    evaluate_model()
