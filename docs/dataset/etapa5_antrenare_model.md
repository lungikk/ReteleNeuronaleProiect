# 📘 README – Etapa 5: Configurarea si Antrenarea Modelului RN

**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti – FIIR
**Student:** Lungeanu Andrei-Alexandru
**Link Repository GitHub:** [Adauga Link Aici]
**Data:** Ianuarie 2026

---

## Scopul Etapei 5

Aceasta etapa se concentreaza pe implementarea scripturilor de antrenare si evaluare pentru modelul RN definit in etapa anterioara. Am finalizat scrierea codului sursa necesar pentru antrenare (`train.py`), evaluare (`evaluate.py`) si definirea arhitecturii (`model.py`), pregatind sistemul pentru executia efectiva a antrenamentului.

---

## PREREQUISITE – Verificare Etapa 4

- [x] **State Machine** definit si documentat
- [x] **Contributie 100% date originale** (1500 intrari generate)
- [x] **Modul 1 (Data Logging)** functional
- [x] **Modul 2 (RN)** arhitectura definita in cod
- [x] **Modul 3 (UI/Web Service)** interfata schelet implementata

---

## Pregatire Date pentru Antrenare

Datasetul utilizat este cel generat integral (`asag_simulated_train_data.csv`), continand 1500 de perechi (Raspuns Student - Raspuns Corect).

**Configuratia seturilor de date (implementata in `train.py`):**
* **Train:** Setul principal pentru ajustarea greutatilor.
* **Validation:** Folosit pentru monitorizarea loss-ului in timp real.
* **Test:** Set separat pentru calculul metricilor finale.

---

## Nivel 1 – Configurarea Antrenarii (Implementata)

### Arhitectura Modelului (`src/neural_network/model.py`)
Am implementat o arhitectura hibrida de tip **Siamese Network** care combina:
1.  **Sentence-Transformer (`all-MiniLM-L6-v2`)**: Pentru generarea embeddings-urilor (vectorizarea textului).
2.  **Regression Head (PyTorch)**: O retea neuronala feed-forward care concateneaza vectorii si prezice nota finala (0-5).

### Tabel Hiperparametri (Configurati in cod)

Urmatorii parametri au fost definiti in scriptul `train.py` si vor fi utilizati la rulare:

| **Hiperparametru** | **Valoare Setata** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Learning Rate** | 2e-4 (0.0002) | Valoare conservatoare pentru a asigura o convergenta stabila a stratului de regresie. |
| **Batch Size** | 16 | Optim pentru a procesa cele 1500 de intrari fara a supraincarca memoria. |
| **Numar Epoci** | 10 | Suficient pentru ca modelul sa invete maparea de la similaritatea vectoriala la nota. |
| **Optimizer** | AdamW | Varianta Adam cu Weight Decay, standard pentru modele bazate pe Transformer. |
| **Loss Function** | MSELoss | Mean Squared Error este metrica ideala pentru probleme de regresie (predictie nota). |

---

## Performanta si Metrici

*Sectiune in asteptare. Urmeaza a fi completata dupa rularea scriptului `evaluate.py`.*

* **MSE (Mean Squared Error):** [Urmeaza a fi generat]
* **Pearson Correlation:** [Urmeaza a fi generat]
* **Acuratete (marja 0.5p):** [Urmeaza a fi generat]

**Locatie salvare model:** `models/trained_model.pth` (va fi generat dupa rulare).

---

## Structura Repository-ului (Actualizata Etapa 5)

Am adaugat scripturile de antrenare si evaluare in structura proiectului:
proiect-rn-[Andrei-Lungeanu]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```
## Instructiuni de Rulare a Codului Implementat

Codul este scris si pregatit. Pasii pentru executie sunt:

1.  **Antrenarea Modelului:**
    ```bash
    python src/neural_network/train.py
    ```
    *Acesta va genera `models/trained_model.pth` si `results/training_history.csv`.*

2.  **Evaluarea Performantei:**
    ```bash
    python src/neural_network/evaluate.py
    ```
    *Acesta va genera graficul `docs/loss_curve.png` si metricile JSON.*

3.  **Testarea in Aplicatie:**
    ```bash
    python src/app/main.py
    ```

---

## Checklist Stare Etapa 5

### Implementare Cod (Realizat)
- [x] Script `model.py` creat (Definire clasa Transformer)
- [x] Script `train.py` creat (Bucla de antrenare, salvare model)
- [x] Script `evaluate.py` creat (Calcul metrici, generare grafice)
- [x] Script `main.py` actualizat (Logica de incarcare model)
- [x] Tabel hiperparametri completat in README

### Executie si Rezultate (Urmeaza a fi realizat)
- [ ] Rulare efectiva `train.py` (Generare fisier .pth)
- [ ] Obtinere metrici finale (Acuratete > 65%)
- [ ] Generare grafic Loss Curve
- [ ] Analiza erorilor pe baza rezultatelor
- [ ] Screenshot inferenta reala in UI

---