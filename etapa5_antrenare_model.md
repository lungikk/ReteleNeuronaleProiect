# 📘 README - Etapa 5: Configurarea si Antrenarea Modelului RN

**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti - FIIR
**Student:** Lungeanu Andrei-Alexandru
**Link Repository GitHub:** https://github.com/lungikk/ReteleNeuronaleProiect.git
**Data predarii:** [08.01.2026]

---

## Scopul Etapei 5

Aceasta etapa corespunde punctului **6. Configurarea si antrenarea modelului RN** din lista de 9 etape.

**Obiectiv principal:** Antrenarea efectiva a modelului RN (MLP Regressor) definit in Etapa 4, evaluarea performantei pe text si integrarea in aplicatia Web Streamlit.

**Pornire obligatorie:** Arhitectura completa si functionala din Etapa 4:
- State Machine definit (Acquisition -> Preprocessing -> Inference -> Grading)
- Cele 3 module functionale (Dataset Loading, TF-IDF+MLP, Streamlit UI)
- Minimum 40% date originale (Generare prin parafrazare/augmentare)

---

## PREREQUISITE - Verificare Etapa 4 (OBLIGATORIU)

**Inainte de a incepe Etapa 5, s-a verificat existenta livrabilelor din Etapa 4:**

- [x] **State Machine** definit si documentat in `docs/state_machine.*`
- [x] **Contributie >=40% date originale** in `data/generated/` (Dataset extins la 1500 exemple prin metode generative)
- [x] **Modul 1 (Data Logging)** functional - CSV-uri structurate (answer_student, answer_correct, score)
- [x] **Modul 2 (RN)** cu arhitectura definita (`models/untrained_model.pkl` - placeholder)
- [x] **Modul 3 (UI/Web Service)** functional in Streamlit
- [x] **Tabelul "Nevoie -> Solutie -> Modul"** complet in README Etapa 4

---

## Pregatire Date pentru Antrenare

### Date noi adaugate in Etapa 4 (contributia de 40%):

S-a refacut preprocesarea pe dataset-ul COMBINAT (Original + Augmentat):

# Scripturi rulate pentru pregatire:
python src/preprocessing/combine_datasets.py
python src/preprocessing/data_cleaner.py  # Lowercase, eliminare punctuatie
python src/preprocessing/feature_engineering.py # Aplicare TF-IDF (1000 features)
python src/preprocessing/data_splitter.py --stratify --random_state 42

Parametri de preprocesare mentinuti:
Vectorizer: TF-IDF (config/vectorizer.pkl) antrenat pe tot corpusul.
Split: 70% train / 15% validation / 15% test.
Random State: 42.

Cerinte Structurate pe 3 Niveluri
Nivel 1 - Obligatoriu pentru Toti (70% din punctaj)
Antrenare model: MLPRegressor antrenat pe 1500 exemple text.

Epoci: Max 300 (oprit automat la 118).

Impartire: 70/15/15.

Metrici test set:

Acuratete: 92.44% (Target >= 65%)

F1-score: 0.9263 (Target >= 0.60)

Salvare model: models/trained_model.pkl (Scikit-Learn format).

Integrare UI: Streamlit incarca modelul antrenat si face predictii reale.

Tabel Hiperparametri si Justificari (OBLIGATORIU - Nivel 1)
Hiperparametru|Valoare Aleasa|Justificare
Algoritm|MLPRegressor|Perceptron Multi-Strat, capabil sa invete relatii non-lineare complexe intre vectorii TF-IDF si note.
Hidden Layers|(100, 50)|Arhitectura cu 2 straturi ascunse (""Funnel shape"") pentru a comprima si extrage trasaturi semantice din cei 1000 de neuroni de input.
Learning Rate|Adaptive (Start 0.001)|Folosit cu solverul SGD. Permite scaderea ratei cand loss-ul stagneaza, asigurand convergenta fina.
Batch size|32|Echilibru optim CPU/Memorie pentru N=1050 samples de antrenare. Asigura stabilitatea gradientului.
Number of epochs|Max 300 (Stop 118)|Early Stopping setat cu patience=5 pentru a preveni overfitting-ul.
Optimizer (Solver)|SGD|Stochastic Gradient Descent, necesar pentru functionalitatea de learning_rate='adaptive'.
Activation|ReLU|Rectified Linear Unit previne saturatia gradientilor si accelereaza antrenarea pe date sparse.
Loss function|MSE (Mean Squared Error)|Fiind o problema de regresie (nota 0.0 - 5.0), MSE penalizeaza erorile mari mai drastic decat MAE.

Nivel 2 - Recomandat (85-90% din punctaj)
Au fost implementate toate optimizarile:

Early Stopping: Antrenarea s-a oprit la epoca 118 deoarece val_loss nu a mai scazut timp de 5 epoci.

Learning Rate Scheduler: Setat pe adaptive. Log-urile arata scaderea LR de la 0.002 la 0.000001 spre finalul antrenarii.

Augmentari relevante domeniu:

Generativa: Dataset extins prin parafrazari automate.

Zgomot Gaussian: Adaugat zgomot (sigma=0.005) peste vectorii TF-IDF la antrenare pentru a forta modelul sa nu memoreze valori exacte (Robustness).

Grafice: Curba de invatare salvata in docs/loss_curve.png.

Analiza erori: Detaliata in sectiunea de mai jos.

Indicatori obtinuti:

Acuratete: 92.44% (Target >= 75%)

F1-score: 0.9263 (Target >= 0.70)

Nivel 3 - Bonus (pana la 100%)
1. Comparare Arhitecturi (Benchmark)
Am antrenat un model clasic (Random Forest) pentru a compara performanta cu RN (MLP).
Model|Acuratete|F1-Score|Timp Predictie|Concluzie
MLP (Retea Neuronala)|92.44%|0.9263|~0.002s|Performanta excelenta si viteza superioara
Random Forest|88.50%|0.8910|~0.015s|Bun, dar fisierul modelului este mare si inferenta mai lenta

Justificare Alegere Finala: Am pastrat MLP deoarece ofera cel mai bun F1-Score (balans precizie/recall) si este extrem de rapid pentru aplicatia Web.

2. Confusion Matrix si Analiza Erori
Matricea de confuzie (docs/confusion_matrix.png) si analiza detaliata a celor 5 exemple gresite sunt incluse in sectiunea "Analiza Erori" de mai jos.

Verificare Consistenta cu State Machine
Fluxul de date respecta arhitectura definita:
Stare din Etapa 4|Implementare in Etapa 5
ACQUIRE_DATA|Input text din Streamlit (Raspuns Student + Barem)
PREPROCESS|Curatare text + Vectorizare TF-IDF (vectorizer.pkl)
RN_INFERENCE|Forward pass prin trained_model.pkl -> Output float
THRESHOLD_CHECK|Rotunjire la grila (0, 2.5, 4.0, 5.0)
ALERT|Feedback vizual in UI (Baloane pentru 5.0, Warning pentru <2.5)

In src/app/web_app.py:
# Modelul este incarcat cu cache pentru performanta
@st.cache_resource
def load_brain():
    model = joblib.load('models/trained_model.pkl') # Weights antrenate
    return model

Analiza Erori in Context Industrial (OBLIGATORIU Nivel 2 & 3)
1. Pe ce clase greseste cel mai mult modelul?
Analiza Matrice de Confuzie: Modelul prezinta dificultati minore in zona notelor mari, specific intre Nota 4.0 (Raspuns Parafrazat) si Nota 5.0 (Raspuns Identic).

Confuzie Principala: Aproximativ 7-8% din raspunsurile perfecte sunt clasificate conservator ca fiind de nota 4.0.

Cauza: Modelul TF-IDF penalizeaza lipsa cuvintelor exacte din barem, chiar daca sensul este corect.

2. Ce caracteristici ale datelor cauzeaza erori?
Lungimea raspunsului (Feature Sparsity): Raspunsurile foarte scurte (1-2 cuvinte) genereaza vectori cu putine informatii.

Sinonime OOV (Out of Vocabulary): Daca studentul foloseste un sinonim care nu a existat in setul de antrenare (ex: "eroare" vs "greseala"), distanta vectoriala creste artificial.

3. Analiza Top 5 Exemple Gresite (Bonus Nivel 3)
Am extras manual cele mai mari discrepante. Toate urmeaza acelasi tipar: Real 5.0 vs AI 4.0.
ID|Raspuns Student|Real|AI|Cauza
Q38|"Long Short-Term Memory, un tip de RNN..."|5.0|4.0|Definitie corecta dar structurata diferit de barem
Q05|"Transforma orice valoare... intre 0 si 1."|5.0|4.0|Lipsa unor termeni tehnici specifici din barem
Q09|"ultimul strat care produce predictia finala..."|5.0|4.0|Explicatie valida, dar considerata parafrazare
Q26|"In Supervised avem etichete..."|5.0|4.0|Similaritate semantica buna, dar nu perfecta lexical
Q23|"Cand modelul este prea simplu..."|5.0|4.0|Scurtimea raspunsului a afectat scorul TF-IDF

Implicatii Industriale:

False Negatives (Impact UX): Studentul primeste 4.0 in loc de 5.0. Este frustrant, dar sigur din punct de vedere academic (nu se acorda note maxime pe nedrept).

4. Ce masuri corective propuneti?
Pentru versiunea 2.0 a produsului EdTech:

Word Embeddings: Trecerea la BERT/RoBERT-a pentru a capta semantica si a rezolva problema sinonimelor.

Human-in-the-loop: Raspunsurile cu scor de incredere la limita (intre 4.0 si 5.0) sa fie marcate pentru revizuire manuala rapida.

Augmentare Sinonime: Generarea automata a mai multor date de antrenare folosind un dictionar de sinonime specific domeniului tehnic.

Structura Repository-ului la Finalul Etapei 5

proiect-rn-[Lungeanu-Andrei]/
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

Instructiuni de Rulare
1. Instalare Dependinte
pip install -r requirements.txt

2. Antrenare Model (Nivel 2)
Ruleaza scriptul care aplica augmentarea si antreneaza reteaua:
python src/neural_network/train_final.py
# Output: Antrenare finalizata. Grafic salvat.

3. Evaluare si Analiza (Nivel 3)
Genereaza metricile si analiza celor 5 erori:
python src/neural_network/evaluate.py
# Output: Acuratete: 92.44% | TOP 5 CELE MAI MARI GRESELI...

4. Lansare Aplicatie Web
Porneste interfata grafica pentru demonstratie:
streamlit run src/app/web_app.py

Checklist Final - Predare
[x] Prerequisite: State Machine, Dataset Augmentat, Module functionale.

[x] Nivel 1: Model antrenat (1500 samples), Metrici >65% (Obtinut 92%), UI Functional.

[x] Nivel 2: Early Stopping, Adaptive LR, Augmentare Zgomot Gaussian, Grafice Loss.

[x] Nivel 3: Comparare MLP vs Random Forest, Analiza detaliata erori (Top 5).

[x] Tehnic: Structura corecta, cod curat, path-uri relative.

Concluzie: Sistemul ASAG (Automated Short Answer Grading) este functional, robust si gata de utilizare demonstrativa, atingand o acuratete de 92.44% pe setul de testare.

