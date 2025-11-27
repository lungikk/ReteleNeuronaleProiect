import pandas as pd
import os
import re

# --- CONFIGURARE ---
INPUT_FILE = 'data/raw/asag_simulated_train_data.csv'
OUTPUT_FILE = 'data/processed/processed.csv'

print(">>> Începe verificarea și procesarea datelor...")

# 1. ÎNCĂRCARE DATE RAW
if not os.path.exists(INPUT_FILE):
    print(f"EROARE CRITICĂ: Nu găsesc fișierul {INPUT_FILE}")
    exit()

df = pd.read_csv(INPUT_FILE)
total_initial = len(df)
print(f"Intrări total încărcate: {total_initial}")

# 2. VERIFICARE CONFORMITATE (Eliminare Nule/Goale)
# Verificăm coloanele esențiale: question_text, answer_correct, answer_student

# Pas A: Eliminare valori NaN (Not a Number / Null din Pandas)
df_clean = df.dropna(subset=['question_text', 'answer_correct', 'answer_student'])

# Pas B: Eliminare string-uri goale sau care conțin doar spații
# Convertim la string, tăiem spațiile (strip) și verificăm dacă lungimea > 0
mask_valid = (
    (df_clean['question_text'].astype(str).str.strip().str.len() > 0) &
    (df_clean['answer_correct'].astype(str).str.strip().str.len() > 0) &
    (df_clean['answer_student'].astype(str).str.strip().str.len() > 0)
)
df_clean = df_clean[mask_valid].copy()

rows_removed = total_initial - len(df_clean)
if rows_removed > 0:
    print(f"⚠️ ATENȚIE: Au fost eliminate {rows_removed} intrări neconforme (nule sau goale).")
else:
    print("✅ Toate datele sunt conforme (nu există valori nule).")

# 3. PROCESARE TEXT (NLP Cleaning)
def clean_text(text):
    text = str(text).lower()                  # Litere mici
    text = re.sub(r'[^\w\s]', '', text)       # Elimină punctuația
    text = re.sub(r'\s+', ' ', text).strip()  # Elimină spații duble
    return text

print("Se aplică curățarea textului (lowercase, punctuatie)...")
df_clean['question_text_cleaned'] = df_clean['question_text'].apply(clean_text)
df_clean['answer_correct_cleaned'] = df_clean['answer_correct'].apply(clean_text)
df_clean['answer_student_cleaned'] = df_clean['answer_student'].apply(clean_text)

# 4. SALVARE DATE PROCESATE
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ SUCCES! Fișierul procesat a fost salvat în: {OUTPUT_FILE}")
print(f"Număr final de intrări valide: {len(df_clean)}")
