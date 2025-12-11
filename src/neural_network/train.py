import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
import os
import time

TRAIN_PATH = 'data/train/train.csv'
VAL_PATH = 'data/validation/validation.csv'
MODEL_SAVE_PATH = 'models/trained_model.pth'
HISTORY_SAVE_PATH = 'results/training_history.csv'

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-4  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Antrenarea va rula pe: {device}")

class ASAGDataset(Dataset):
    """
    Clasă care transformă CSV-ul în format compatibil PyTorch.
    Returnează: (Răspuns Student, Răspuns Corect, Notă)
    """
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)[['answer_student', 'answer_correct', 'score_manual']]
        # Convertim totul la string pentru siguranță și nota la float
        self.data['answer_student'] = self.data['answer_student'].astype(str)
        self.data['answer_correct'] = self.data['answer_correct'].astype(str)
        self.data['score_manual'] = self.data['score_manual'].astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['answer_student'], row['answer_correct'], torch.tensor(row['score_manual'], dtype=torch.float32)

class ASAGTransformerModel(nn.Module):
    """
    Arhitectura hibridă:
    1. Sentence Transformer (Embeddings)
    2. Concatenare vectori
    3. Strat DENSE (Regresie) pentru nota finala
    """
    def __init__(self):
        super(ASAGTransformerModel, self).__init__()
        # Încărcăm modelul pre-antrenat SBERT (il inghetam pentru viteza pe CPU)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  

        # Input: 384 (student) + 384 (corect) = 768
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

        combined_features = torch.cat((u, v), dim=1)

        output = self.regressor(combined_features)
        
        return output.squeeze()

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch_idx, (stud_text, corr_text, scores) in enumerate(dataloader):
        scores = scores.to(device)
        
        optimizer.zero_grad()
        
        predictions = model(list(stud_text), list(corr_text))
        
        loss = criterion(predictions, scores)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for stud_text, corr_text, scores in dataloader:
            scores = scores.to(device)
            predictions = model(list(stud_text), list(corr_text))
            loss = criterion(predictions, scores)
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    print(f"--- Începe Antrenarea ---")
    
    # Creare directoare dacă nu există
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Încărcare Date
    train_dataset = ASAGDataset(TRAIN_PATH)
    val_dataset = ASAGDataset(VAL_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"Date antrenament: {len(train_dataset)} | Date validare: {len(val_dataset)}")

    model = ASAGTransformerModel().to(device)
    
    criterion = nn.MSELoss() # Mean Squared Error (pentru regresie)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = []

    start_time = time.time()

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate(model, val_loader, criterion)
        
        history.append([epoch+1, train_loss, val_loss])
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Salvare cel mai bun model (Checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # print("  -> Model salvat (performanță îmbunătățită)")

    total_time = time.time() - start_time
    print(f"\n✅ Antrenare finalizată în {total_time:.2f} secunde.")
    print(f"Modelul antrenat este salvat în: {MODEL_SAVE_PATH}")

    hist_df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_loss'])
    hist_df.to_csv(HISTORY_SAVE_PATH, index=False)
    print(f"Istoricul antrenării salvat în: {HISTORY_SAVE_PATH}")
