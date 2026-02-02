import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class ASAGTransformerModel(nn.Module):
    def __init__(self, frozen_bert=True):
        super(ASAGTransformerModel, self).__init__()
        
        print("Se incarca modelul SentenceTransformer...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        self.frozen_bert = frozen_bert

        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, student_texts, correct_texts):
        device = next(self.parameters()).device
        
        if self.frozen_bert:
            with torch.no_grad():
                u = self.encoder.encode(student_texts, convert_to_tensor=True, device=device)
                v = self.encoder.encode(correct_texts, convert_to_tensor=True, device=device)
        else:
            u = self.encoder.encode(student_texts, convert_to_tensor=True, device=device)
            v = self.encoder.encode(correct_texts, convert_to_tensor=True, device=device)

        combined_features = torch.cat((u, v), dim=1)
        score = self.regressor(combined_features)
        
        return score.squeeze()

if __name__ == "__main__":
    print("--- Testare Arhitectura Model ---")
    
    try:
        model = ASAGTransformerModel()
        print("Model creat cu succes.")
        
        student = ["O retea neuronala este un model inspirat biologic."]
        barem = ["Retelele neuronale sunt sisteme de calcul inspirate din creier."]
        
        print("Se testeaza o predictie...")
        score = model(student, barem)
        
        print(f"Predictie reusita! Scor dummy: {score.item():.4f}")
        
    except Exception as e:
        print(f"Eroare la initializare: {e}")
