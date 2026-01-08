import pandas as pd
import os
from sklearn.model_selection import train_test_split

PROJECT_ROOT = r"C:\FACULTATE\ANUL 3 SEM 1\RN"

RAW_FILE = os.path.join(PROJECT_ROOT, 'data', 'raw', 'asag_simulated_train_data.csv')
TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'validation')
TEST_DIR = os.path.join(PROJECT_ROOT, 'data', 'test')

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def proceseaza_date():
    print(f" Caut fisierul brut la: {RAW_FILE}")

    if not os.path.exists(RAW_FILE):
        print(f"\n EROARE: Nu gasesc fisierul!")
        print(f"Te rog verifica daca ai folderul 'data/raw' si fisierul csv in: {PROJECT_ROOT}")
        return

    print(" Fisier gasit! Incep procesarea...")
    df = pd.read_csv(RAW_FILE)

    print(f"   Total intrari incarcate: {len(df)}")

    df.dropna(subset=['answer_student', 'answer_correct'], inplace=True)

    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df['score_manual']
    )

    val_size_adjusted = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=42, stratify=train_val_df['score_manual']
    )

    train_df.to_csv(os.path.join(TRAIN_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(VAL_DIR, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, 'test.csv'), index=False)

    print("\n Datele au fost impartite:")
    print(f"   Train: {len(train_df)} randuri -> salvat in data/train")
    print(f"   Validation: {len(val_df)} randuri -> salvat in data/validation")
    print(f"   Test: {len(test_df)} randuri -> salvat in data/test")


if __name__ == "__main__":
    proceseaza_date()
