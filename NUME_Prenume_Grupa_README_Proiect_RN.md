## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | [Lungeanu Andrei-Alexandru] |
| **Grupa / Specializare** | [634AB / Informatică Industrială] |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | [https://github.com/lungikk/ReteleNeuronaleProiect.git] |
| **Acces Repository** | [Public] |
| **Stack Tehnologic** | [Python (Scikit-Learn, Pandas, Streamlit)] |
| **Domeniul Industrial de Interes (DII)** | [Educatie / EdTech (Automatizarea Evaluarii)] |
| **Tip Rețea Neuronală** | [MLP (Multi-Layer Perceptron - Regressor)] |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | [85%] | [92.44%] | [+7.44%] | [✓] |
| F1-Score (Macro) | ≥0.65 | [0.83] | [0.9263] | [+0.0963] | [✓] |
| Latență Inferență | [≤50 ms] | [10 ms] | [2 ms] | [-8 ms] | [✓] |
| Contribuție Date Originale | ≥40% | [40%] | [40%] | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | [4] | [4] | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință                                                                 | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1   | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA     |
| 2   | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA     |
| 3   | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA     |
| 4   | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA     |
| 5   | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA     |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

*[Descrieți în 1-2 paragrafe: Ce problemă concretă din domeniul industrial rezolvă acest proiect? Care este contextul și situația actuală? De ce este importantă rezolvarea acestei probleme?]*

In contextul expansiunii accelerate a platformelor de e-learning si a cresterii numarului de studenti in mediul universitar, evaluarea manuala a raspunsurilor deschise (text liber) a devenit un blocaj operational major. Procesul traditional este extrem de consumator de timp, genereaza intarzieri in oferirea notelor si este inevitabil afectat de factorul uman: subiectivism, oboseala si mai ales inconsistenta in aplicarea baremului pe loturi mari de lucrari.

Acest proiect propune o solutie de tip ASAG (Automated Short Answer Grading) care automatizeaza procesul de verificare folosind Retele Neuronale. Importanta industriala a solutiei rezida in scalabilitate si obiectivitate: sistemul poate evalua mii de raspunsuri instantaneu, oferind feedback in timp real studentilor si eliminand variatiile de notare. Astfel, profesorii sunt scutiti de munca repetitiva, putandu-se concentra pe mentorat, in timp ce institutiile reduc costurile operationale si cresc calitatea actului educational.

### 2.2 Beneficii Măsurabile Urmărite

*[Listați 3-5 beneficii concrete cu metrici țintă]*

1. [Reducerea timpului de evaluare cu peste 90%: Trecerea de la minute per lucrare (manual) la milisecunde per raspuns (automat), permitand procesarea unor volume mari de date instantaneu.]
2. [Acuratete comparabila cu nivelul uman (> 85%): Sistemul tinteste o precizie ridicata in alinierea cu baremul profesorului, minimizand erorile de notare (target atins: 92.44%).]
3. [Eliminarea subiectivitatii (Consistenta 100%): Garantarea faptului ca acelasi raspuns primeste intotdeauna aceeasi nota, indiferent de momentul zilei sau de starea evaluatorului.]
4. [Feedback instantaneu: Reducerea timpului de asteptare pentru studenti de la zile/saptamani la mai putin de 1 secunda.]
5. [Reducerea costurilor operationale: Diminuarea efortului manual necesar din partea personalului didactic pentru activitati repetitive de corectare.]

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Reducerea timpului masiv de corectare manuala a testelor scrise (estimat la 30-60 min/test) | Evaluare automata instanta a raspunsurilor textuale -> nota generata in < 5 secunde/raspuns | Neural Network + Scoring Module |
| Eliminarea subiectivitatii si inconsistentei in notarea raspunsurilor deschise (eroare umana ~15%) | Calcularea scorului de similaritate semantica fata de barem cu o acuratete estimata de > 85% | Preprocessing + Neural Network (Transformer) |
| Gestionarea volumului mare de studenti si necesitatea feedback-ului rapid | Procesarea simultana a cererilor si stocarea rezultatelor pentru 1000+ studenti fara intarzieri | Web Service + Data Logging |
---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | [Mixt (Dataset Public + Augmentare Sintetica)] |
| **Sursa concretă** | [Mohler Dataset (University of North Texas) - Computer Science] |
| **Număr total observații finale (N)** | [1500] |
| **Număr features** | [1000 (Vectori TF-IDF rezultati din text)] |
| **Tipuri de date** | [Text (NLP)] |
| **Format fișiere** | [CSV] |
| **Perioada colectării/generării** | [Noiembrie 2025 - Ianuarie 2026] |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | [1500] |
| **Observații originale (M)** | [600] |
| **Procent contribuție originală** | [40%] |
| **Tip contribuție** | [Date sintetice] |
| **Locație cod generare** | `src/data_acquisition.py` |
| **Locație date originale** | `data/generated/` |

**Descriere metodă generare/achiziție:**

*[Explicați în 1-2 paragrafe: Cum ați generat/achiziționat datele originale? Ce parametri ați folosit? De ce sunt relevante pentru problema voastră?]*

Datele originale au fost generate programatic utilizand un script dedicat de Data Augmentation (src/data_acquisition/). Metoda principala a constat in aplicarea tehnicii de substitutie a sinonimelor si parafrazare controlata asupra raspunsurilor din setul Mohler. Concret, pentru fiecare raspuns "sursa", s-au generat variatii lexicale prin inlocuirea termenilor tehnici cu sinonime uzuale sau modificarea topicii, fara a altera sensul corect al raspunsului sau punctajul asociat.

Aceasta abordare este esentiala pentru robustetea sistemului ASAG. Deoarece dataset-ul initial era dezechilibrat si limitat ca vocabular, augmentarea a servit un dublu scop: (1) echilibrarea claselor (generand mai multe exemple pentru notele intermediare de 2.5 si 4.0) si (2) prevenirea overfitting-ului, fortand reteaua MLP sa invete relatii semantice (intelesul) in loc sa memoreze secvente specifice de cuvinte.

### 3.3 Preprocesare și Split Date

| Set | Procent | Număr Observații |
|-----|---------|------------------|
| Train | 70% | [1050] |
| Validation | 15% | [225] |
| Test | 15% | [225] |

**Preprocesări aplicate:**
- [Normalizare Text: Conversie la litere mici (lowercase), eliminare punctuatie si caractere speciale irelevante pentru a reduce zgomotul]
- [Vectorizare (TF-IDF): Transformarea textului in vectori numerici de dimensiune fixa, limitand vocabularul la cele mai relevante 1000 de cuvinte (max_features=1000)]
- [Ingineria Trasaturilor (Feature Engineering): Concatenarea [Raspuns Student] + [Barem] inainte de vectorizare pentru a oferi context semantic retelei]
- [Stratificare: Impartirea datelor s-a realizat folosind stratify=y, garantand ca proportia notelor (0.0, 2.5, 4.0, 5.0) este identica in toate cele trei seturi (Train, Val, Test)]

**Referințe fișiere:** `data/README.md`, `models/vectorizer.pkl`

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **1. Data Logging / Acquisition** | Python (`pandas`, `random`) | Functional. Genereaza CSV cu 1500 intrari. |
| **2. Neural Network Module** | Python (`sentence-transformers`) | Definit. Modelul Transformer este incarcat si functional pentru inferenta (embedding). |
| **3. Web Service / UI** | Python (CLI Demo / `input()`) | Functional. Permite introducerea unui raspuns si afiseaza nota. |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png` 

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | [Asteptare interactiune utilizator in UI] | [Start aplicatie Streamlit] | [Selectare mod lucru] |
| `ACQUIRE_DATA` | [Preluare input (Text manual sau Load CSV)] | [Buton "Genereaza"] | [Date validate] |
| `PREPROCESS` | [Curatare text si Vectorizare TF-IDF] | [Text brut disponibi] | [Vectori numerici (1000 features)] |
| `INFERENCE` | [Forward pass prin MLP Regressor] | [Input vectorizat disponibil] | [Scor brut (float) generat] |
| `DECISION` | [Discretizare scor brut la grila (0, 2.5, 4, 5)] | [Output RN disponibil] | [Nota finala stabilita] |
| `OUTPUT/ALERT` | [Afisare nota si feedback vizual] | [Decizie luată] | [Asteptare input nou] |
| `ERROR` | [Afisare mesaj eroare (ex: text prea scurt)] | [Exceptie / Input invalid] | [Revenire la IDLE] |

**Justificare alegere arhitectură State Machine:**

*[1 paragraf: De ce această structură pentru problema voastră specifică?]*

Arhitectura bazata pe State Machine a fost aleasa pentru a gestiona robust cele doua fluxuri de utilizare implementate in Etapa 6: evaluarea punctuala (Manual) si evaluarea in lot (Chestionar). Aceasta structura garanteaza secventialitatea obligatorie a pasilor (un text nu poate fi notat fara a fi preprocesat identic cu setul de antrenare) si permite izolarea erorilor, asigurand ca o problema la incarcarea unui fisier CSV nu blocheaza intreaga aplicatie, ci doar tranzitioneaza sistemul intr-o stare de eroare controlata, protejand experienta utilizatorului.

### 4.3 Actualizări State Machine în Etapa 6 (dacă este cazul)

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
|Model Utilizat|trained_model.pkl|optimized_model.pkl|Acuratete superioara (92.44%) si robustete la sinonime prin augmentare|
|Flux Stari (Logica)|Liniar (Single Input)|Ramificat (Selector Mod)|Adaugarea functionalitatii de ""Generator Chestionare"" pentru testare rapida|
|Stare Noua Adaugata|N/A|BATCH_PROCESSING|Procesarea simultana a 5 intrebari extrase aleatoriu din setul de test|
|Feedback Vizual (UI)|Text simplu|Tabel Colorat (Verde/Rosu)|Identificarea vizuala instanta a erorilor majore vs minore|
|Latenta Target|~10ms / input|~2ms / input|Optimizare necesara pentru a rula inferenta pe loturi (batch) fara lag|
---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

```
Input (shape: [1000]) - Vectori TF-IDF (Bag of Words)
  → Dense(100, Activation: ReLU)  - Strat Ascuns 1
  → Dense(50,  Activation: ReLU)  - Strat Ascuns 2 (Structura Funnel)
  → Output(1,  Activation: Identity) - Strat Final Regresie
Output: 1 valoare continua (Nota estimata 0.0 - 5.0)
```

**Justificare alegere arhitectură:**

*[1-2 propoziții: De ce această arhitectură? Ce alternative ați considerat și de ce le-ați respins?]*

Am optat pentru o arhitectura MLP (Multi-Layer Perceptron) de tip "Funnel" (1000 -> 100 -> 50) deoarece permite compresia treptata a informatiilor din vectorii rari (sparse) TF-IDF in trasaturi semantice abstracte, necesare pentru a invata relatiile non-lineare dintre cuvinte si nota. Am respins modelele simple (Regresie Lineara) din cauza underfitting-ului si modelele complexe (BERT/Transformers) deoarece ar fi incalcat cerinta de latenta scazuta (<50ms) fara hardware specializat (GPU).

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | ['adaptive'] | [Ajustare dinamica pentru a iesi din minime locale (folosit cu SGD)] |
| Batch Size | [32] | [Compromis optim intre viteza de calcul si stabilitatea gradientului] |
| Epochs | [118 (Max 300)] | [Early Stopping activat: antrenarea s-a oprit automat cand loss-ul a stagnat] |
| Optimizer | [SGD] | [Stochastic Gradient Descent s-a dovedit mai robust decat Adam pe acest dataset mic] |
| Loss Function | [Mean Squared Error (MSE)] | [Modelul este un Regressor (prezice valori continue), nu un Clasificator] |
| Regularizare | [L2 (Alpha=0.0001)] | [Penalizare standard pentru a mentine greutatile mici + Augmentare Zgomot] |
| Early Stopping | [n_iter_no_change=10] | [Prevenire overfitting: oprire daca scorul de validare nu creste timp de 10 epoci] |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy | F1-Score | Timp Antrenare | Observații |
|------|----------------------------|----------|----------|----------------|------------|
| **Baseline** | Configurația din Etapa 5 | [85%] | [0.83] | [1 min] | Referință |
| Exp 1 | [Arhitectura Funnel (100, 50)] | [88%] | [0.86] | [1.5 min] | [Structura mai adanca extrage trasaturi semantice mai bune] |
| Exp 2 | [Schimbare Solver: Adam -> SGD] | [89%] | [0.88] | [2 min] | [Convergenta mai lenta dar mai stabila, generalizare mai buna] |
| Exp 3 | [Augmentare Zgomot Gaussian] | [92.44%] | [0.9263] | [2.5 min] | [Salt major: modelul devine rezistent la sinonime neintalnite] |
| **FINAL** | [Exp 3 + Early Stopping] | **[92.44%]** | **[0.9263]** | [2.5 min] | **Modelul folosit în producție** |

**Justificare alegere model final:**

*[1 paragraf: De ce această configurație? Ce compromisuri ați făcut între accuracy/timp/complexitate?]*

Configuratia finala (MLP cu arhitectura Funnel + SGD + Augmentare cu Zgomot) a fost aleasa deoarece a oferit cel mai bun echilibru intre capacitatea de generalizare si viteza de inferenta. Desi un model Transformer (BERT) ar fi putut atinge o acuratete marginal mai mare, acesta ar fi crescut timpul de raspuns la sute de milisecunde. Modelul nostru MLP atinge o acuratete de peste 92% cu un timp de raspuns de doar 2ms, fiind ideal pentru integrarea intr-o platforma educationala real-time unde viteza este critica.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.pkl`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | [92.44%] | ≥70% | [✓] |
| **F1-Score (Macro)** | [0.9263] | ≥0.65 | [✓] |
| **Precision (Macro)** | [0.9250] | - | - |
| **Recall (Macro)** | [0.9270] | - | - |

**Îmbunătățire față de Baseline (Etapa 5):**

| Metric | Etapa 5 (Baseline) | Etapa 6 (Optimizat) | Îmbunătățire |
|--------|-------------------|---------------------|--------------|
| Accuracy | [85%] | [92.44%] | [+7.44%] |
| F1-Score | [0.83] | [0.9263] | [+0.0963] |

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**

| Aspect | Observație |
|--------|------------|
| **Clasa cu cea mai bună performanță** | Nota 0.0 (Incorect) - Precision 98%, Recall 99%. Modelul identifica excelent raspunsurile gresite sau irelevante |
| **Clasa cu cea mai slabă performanță** | Nota 4.0 - Precision 88%, Recall 85%. Este zona de mijloc, sensibila la nuante fine |
| **Confuzii frecvente** | 4.0 confundat cu 5.0. Modelul tinde sa depuncteze raspunsurile corecte care folosesc formulari atipice (sinonime rare) |
| **Dezechilibru clase** | Clasele extreme (0.0 si 5.0) au fost usor de invatat, in timp ce clasele intermediare au necesitat augmentare masiva pentru a atinge performanta actuala |

### 6.3 Analiza Top 5 Erori

| # | Input (descriere scurtă) | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|--------------------------|--------------|-------------|-----------------|------------------------|
|1|Raspuns corect folosind sinonimul "stiva" in loc de "stack"|4.0|5.0|Limitare TF-IDF: Cuvantul "stiva" nu a aparut des in antrenare, deci scorul de similaritate a scazut.|Acceptabila: Studentul va face contestatie, iar profesorul va corecta. E preferabil fata de a da nota maxima eronat|
|2|Raspuns partial corect dar foarte scurt (3 cuvinte)|2.5|4.0|Lipsa Context: Numarul mic de cuvinte a generat un vector rar (sparse) cu putina informatie|Risk Mediu: Necesita validare umana pentru raspunsurile sub 5 cuvinte|
|3|Raspuns lung cu multe cuvinte cheie, dar fara logica (Word Salad)|5.0|0.0|Lipsa Intelegere Secventiala: MLP (BoW) vede cuvintele, dar nu ordinea lor. A vazut keyword-urile si a dat nota|Risk Critic: Studentii pot "pacali" sistemul scriind cuvinte cheie la intamplare. Solutie viitoare: BERT|
|4|Negatie subtila (""NU este o structura..."")|4.0|0.0|Ignorare Negatii: TF-IDF trateaza adesea "nu" ca stop-word sau ii da importanta mica|False Positive: Sistemul puncteaza o afirmatie gresita ca fiind corecta|
|5|Raspuns corect dar cu greseli gramaticale grave|2.5|5.0|Zgomot in Input: Cuvintele gresite ("algoritm" scris "algorim") nu au facut match in vocabular|Educational: Incurajeaza studentii sa scrie corect gramatical|

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**

*[1 paragraf: Traduceți metricile în impact real în domeniul vostru industrial]*

Cu o acuratete de 92.44%, sistemul actioneaza ca un filtru extrem de eficient. Dintr-un lot de 1.000 de lucrari, profesorul trebuie sa verifice manual doar aproximativ 70-80 de cazuri (cele cu scoruri incerte sau contestatii), in loc sa corecteze 1.000. Modelul adopta o strategie "Conservative Fail-Safe": in caz de dubiu, tinde sa acorde o nota mai mica (4.0 in loc de 5.0). In industrie, acest lucru reduce riscul de a acorda diplome/credite nemeritate. Costul reinspectiei (pentru contestatii) este mult mai mic decat costul reputational al unei evaluari incorecte "in sus".

**Pragul de acceptabilitate pentru domeniu:** [Acuratete ≥ 85% pentru sisteme asistive]  
**Status:** [Atins si Depasit (+7.44%)]  

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.pkl` | `optimized_model.pkl` | [Acuratete crescuta (+7.44%) si robustete la sinonime (Data Augmentation)] |
| **Threshold decizie** | [Output brut (float)] | [Rotunjire la grila (0, 2.5, 4, 5)] | [Aliniere cu sistemul academic de notare (Nearest Neighbor] |
| **UI - feedback vizual** | [Text simplu] | [Tabel Colorat (Verde/Rosu)] | [Validare vizuala instanta pentru profesor (Identificare erori)] |
| **UI - functionalitate** | [Input manual simplu] | [Mod Dual: Manual + Generator Chestionare] | [Posibilitatea de a testa rapid loturi de intrebari aleatoare] |
| **Inference Engine** | [Procesare secventiala] | [Procesare Batch (Loturi)] | [Optimizare viteza pentru seturi mari de date (~2ms/student)] |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

*[Descriere scurtă: Ce se vede în screenshot? Ce demonstrează?]*

Descriere scurta: Screenshot-ul demonstreaza modul "Generator Chestionare" al aplicatiei Streamlit. Se observa un tabel cu 5 intrari extrase aleatoriu din setul de test. Coloanele afiseaza Raspunsul Studentului, Baremul, Nota Reala si Nota Prezisa. Randurile sunt codate cromatic: fundalul verde indica o predictie corecta, iar cel rosu (daca ar exista) ar indica o divergenta intre model si profesor. In bara laterala sunt afisati parametrii modelului optimizat (Acuratete 92.44%).

### 7.3 Demonstrație Funcțională End-to-End

**Locație dovadă:** `docs/demo/` *(Secvență screenshots)*

**Fluxul demonstrat:**

| Pas | Acțiune | Rezultat Vizibil |
|-----|---------|------------------|
| 1 | Input | [Utilizatorul apasa butonul "Genereaza Chestionar Test" in UI] |
| 2 | Procesare | [Sistemul extrage random 5 perechi (Student-Barem), le vectorizeaza si ruleaza inferenta] |
| 3 | Inferență | [Modelul MLP prezice 5 note simultan in mai putin de 0.1 secunde] |
| 4 | Decizie | [Apare tabelul cu rezultate. Codurile de culoare confirma acuratetea predictiilor] |

**Latență măsurată end-to-end:** [50] ms  
**Data și ora demonstrației:** [13.01.2026, 14:51]

---

## 8. Structura Repository-ului Final

```
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
```

### Legendă Progresie pe Etape

| Folder / Fișier | Etapa 3 | Etapa 4 | Etapa 5 | Etapa 6 |
|-----------------|:-------:|:-------:|:-------:|:-------:|
| `data/raw/`, `processed/`, `train/`, `val/`, `test/` | ✓ Creat | - | Actualizat* | - |
| `data/generated/` | - | ✓ Creat | - | - |
| `src/preprocessing/` | ✓ Creat | - | Actualizat* | - |
| `src/data_acquisition/` | - | ✓ Creat | - | - |
| `src/neural_network/model.py` | - | ✓ Creat | - | - |
| `src/neural_network/train.py`, `evaluate.py` | - | - | ✓ Creat | - |
| `src/neural_network/optimize.py`, `visualize.py` | - | - | - | ✓ Creat |
| `src/app/` | - | ✓ Creat | Actualizat | Actualizat |
| `models/untrained_model.*` | - | ✓ Creat | - | - |
| `models/trained_model.*` | - | - | ✓ Creat | - |
| `models/optimized_model.*` | - | - | - | ✓ Creat |
| `docs/state_machine.*` | - | ✓ Creat | - | (v2 opțional) |
| `docs/etapa3_analiza_date.md` | ✓ Creat | - | - | - |
| `docs/etapa4_arhitectura_SIA.md` | - | ✓ Creat | - | - |
| `docs/etapa5_antrenare_model.md` | - | - | ✓ Creat | - |
| `docs/etapa6_optimizare_concluzii.md` | - | - | - | ✓ Creat |
| `docs/confusion_matrix_optimized.png` | - | - | - | ✓ Creat |
| `docs/screenshots/` | - | ✓ Creat | Actualizat | Actualizat |
| `results/training_history.csv` | - | - | ✓ Creat | - |
| `results/optimization_experiments.csv` | - | - | - | ✓ Creat |
| `results/final_metrics.json` | - | - | - | ✓ Creat |
| **README.md** (acest fișier) | Draft | Actualizat | Actualizat | **FINAL** |

*\* Actualizat dacă s-au adăugat date noi în Etapa 4*

### Convenție Tag-uri Git

| Tag | Etapa | Commit Message Recomandat |
|-----|-------|---------------------------|
| `v0.3-data-ready` | Etapa 3 | "Etapa 3 completă - Dataset analizat și preprocesat" |
| `v0.4-architecture` | Etapa 4 | "Etapa 4 completă - Arhitectură SIA funcțională" |
| `v0.5-model-trained` | Etapa 5 | "Etapa 5 completă - Accuracy=85%, F1=0.83" |
| `v0.6-optimized-final` | Etapa 6 | "Etapa 6 completă - Accuracy=92.44%, F1=0.92 (optimizat)" |

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

```
Python >= 3.9
pip >= 21.0
OS: Windows
```

### 9.2 Instalare

```bash
# 1. Clonare repository
git clone [https://github.com/lungikk/ReteleNeuronaleProiect.git]
cd proiect-rn-[Lungeanu-Andrei]

# 2. Creare mediu virtual (recomandat)
python -m venv venv
venv\Scripts\activate    # Windows

# 3. Instalare dependențe
pip install -r requirements.txt
```

### 9.3 Rulare Pipeline Complet

```bash
# Pasul 1: Preprocesare date (dacă rulați de la zero)
python src/preprocessing.py

# Pasul 2: Antrenare model (pentru reproducere rezultate)
python src/neural_network/train.py --config config/optimized_config.yaml

# Pasul 3: Evaluare model pe test set
python src/neural_network/evaluate.py --model models/optimized_model.pkl

# Pasul 4: Lansare aplicație UI
streamlit run src/app/web_app.py
```

### 9.4 Verificare Rapidă 

```bash
# Verificare că modelul se încarcă corect
python -c "from src.neural_network.model import load_model; m = load_model('models/optimized_model.pkl'); print('✓ Model încărcat cu succes')"

# Verificare inferență pe un exemplu
python src/neural_network/evaluate.py --model models/optimized_model.pkl --quick-test

```

---

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit (Secțiunea 2) | Target | Realizat | Status |
|--------------------------------|--------|----------|--------|
| Reducerea timpului de evaluare | < 1 sec/student | [realizat] | [✓] |
| Eliminarea subiectivitatii | 100% Consistent | [realizat] | [✓] |
| Accuracy pe test set | ≥70% | [92.44%] | [✓] |
| F1-Score pe test set | ≥0.65 | [0.9263] | [✓] |
| Robustete la sinonime | Mediu | Ridicat (prin Augmentare) | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

*[Fiți onești - evaluatorul apreciază identificarea clară a limitărilor]*

1. **Limitare Semantica (Word Salad)**: Deoarece folosim TF-IDF (Bag of Words), modelul nu tine cont de ordinea cuvintelor. O fraza fara sens gramatical, dar care contine cuvintele cheie corecte, poate primi nota 5.0 (False Positive).

2. **Gestionarea Negatiilor**: Modelul are dificultati in a distinge intre "Acesta este un algoritm" si "Acesta NU este un algoritm", deoarece cuvantul "nu" are o pondere mica in TF-IDF.

3. **Sensibilitate la Raspunsuri Foarte Scurte**: Raspunsurile formate din 1-2 cuvinte genereaza vectori "rari" (sparse), ceea ce duce uneori la subevaluare (nota 2.5 in loc de 4.0).

4. **Functionalitati planificate dar neimplementate**: Integrarea cu un API de dictionar de sinonime live (in prezent folosim doar augmentarea statica din antrenare).

### 10.3 Lecții Învățate (Top 5)

1. **Calitatea Datelor > Complexitatea Modelului**: Am observat ca curatarea textului si augmentarea datelor au adus un salt de performanta (+7%) mult mai mare decat adaugarea de straturi neuronale suplimentare.

2. **Early Stopping este Critic**: Pe un dataset mic (1500 intrari), modelul tinde sa memoreze datele extrem de rapid. Fara Early Stopping, loss-ul pe validare exploda dupa epoca 120.

3. **Zgomotul Gaussian ca Regularizare**: Adaugarea de variatii aleatoare mici peste vectorii de intrare a fortat reteaua sa invete "conceptul" general, nu valorile fixe, rezolvand problema overfitting-ului.

4. **Interpretarea Erorilor**: Matricea de confuzie ne-a aratat ca modelul este "conservator" (prefera sa depuncteze decat sa supranoteze), ceea ce este un comportament de dorit intr-un sistem educational (Fail-Safe).

5. **Documentarea Iterativa**: Mentinerea fisierelor de log si a istoricului experimentelor a redus timpul de scriere a raportului final cu 50%.

### 10.4 Retrospectivă

**Ce ați schimba dacă ați reîncepe proiectul?**

*[1-2 paragrafe: Decizii pe care le-ați lua diferit, cu justificare bazată pe experiența acumulată]*

Daca as relua proiectul de la zero, as inlocui vectorizarea TF-IDF cu Word Embeddings pre-antrenate (ex: GloVe sau Word2Vec). Desi TF-IDF este rapid, pierde contextul semantic si relatiile dintre cuvinte. Embeddings-urile ar permite modelului sa inteleaga ca "masina" si "automobil" sunt identice fara a fi nevoie de augmentare explicita.

De asemenea, as implementa de la inceput un sistem de Cross-Validation cu 5 fold-uri in loc de un simplu split Train/Test, pentru a avea o certitudine statistica mai mare asupra acuratetei raportate.

### 10.5 Direcții de Dezvoltare Ulterioară

| Termen | Îmbunătățire Propusă | Beneficiu Estimat |
|--------|---------------------|-------------------|
| **Short-term** (1-2 săptămâni) | [Adaugare lista Stop-Words personalizata pentru domeniu] | [Eliminarea zgomotului specific (ex: "in", "la", "care") -> +1-2% Acuratete] |
| **Medium-term** (1-2 luni) | [Migrare la arhitectura Transformer (DistilBERT)] | [Rezolvarea problemelor de negatie si topica a frazei] |
| **Long-term** | [Deployment ca Microserviciu (Docker + REST API)] | [Integrare posibila cu platforme LMS reale (Moodle/Blackboard)] |

---

## 11. Bibliografie

*[Minimum 3 surse cu DOI/link funcțional - format: Autor, Titlu, Anul, Link]*

1. Mohler, M., Bunescu, R., & Mihalcea, R., Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments, Association for Computational Linguistics (ACL), 2011. URL: https://aclanthology.org/P11-1076/ (Sursa Dataset-ului original utilizat in proiect)

2. Pedregosa, F. et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research (JMLR), Vol. 12, pp. 2825-2830, 2011. URL: https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html (Referinta oficiala pentru algoritmii MLPRegressor si TfidfVectorizer utilizati)

3. Burrows, S., Gurevych, I., & Stein, B., The Eras and Trends of Automatic Short Answer Grading, International Journal of Artificial Intelligence in Education, 25, 60–117, 2015. DOI: https://doi.org/10.1007/s40593-014-0026-8 (Studiu de referinta despre evolutia sistemelor ASAG)

4. Streamlit Documentation, Streamlit: The fastest way to build and share data apps, 2024. URL: https://docs.streamlit.io/ (Documentatia pentru interfata grafica a aplicatiei)

**Exemple format:**
- Abaza, B., 2025. AI-Driven Dynamic Covariance for ROS 2 Mobile Robot Localization. Sensors, 25, 3026. https://doi.org/10.3390/s25103026
- Keras Documentation, 2024. Getting Started Guide. https://keras.io/getting_started/

---

## 12. Checklist Final (Auto-verificare înainte de predare)

### Cerințe Tehnice Obligatorii

- [x] **Accuracy ≥70%** pe test set (verificat în `results/final_metrics.json`)
- [x] **F1-Score ≥0.65** pe test set
- [x] **Contribuție ≥40% date originale** (verificabil în `data/generated/`)
- [x] **Model antrenat de la zero** (NU pre-trained fine-tuning)
- [x] **Minimum 4 experimente** de optimizare documentate (tabel în Secțiunea 5.3)
- [x] **Confusion matrix** generată și interpretată (Secțiunea 6.2)
- [x] **State Machine** definit cu minimum 4-6 stări (Secțiunea 4.2)
- [x] **Cele 3 module funcționale:** Data Logging, RN, UI (Secțiunea 4.1)
- [x] **Demonstrație end-to-end** disponibilă în `docs/demo/`

### Repository și Documentație

- [x] **README.md** complet (toate secțiunile completate cu date reale)
- [x] **4 README-uri etape** prezente în `docs/` (etapa3, etapa4, etapa5, etapa6)
- [x] **Screenshots** prezente în `docs/screenshots/`
- [x] **Structura repository** conformă cu Secțiunea 8
- [x] **requirements.txt** actualizat și funcțional
- [x] **Cod comentat** (minim 15% linii comentarii relevante)
- [x] **Toate path-urile relative** (nu absolute: `/Users/...` sau `C:\...`)

### Acces și Versionare

- [ ] **Repository accesibil** cadrelor didactice RN (public sau privat cu acces)
- [ ] **Tag `v0.6-optimized-final`** creat și pushed
- [ ] **Commit-uri incrementale** vizibile în `git log` (nu 1 commit gigantic)
- [ ] **Fișiere mari** (>100MB) excluse sau în `.gitignore`

### Verificare Anti-Plagiat

- [x] Model antrenat **de la zero** (weights inițializate random, nu descărcate)
- [x] **Minimum 40% date originale** (nu doar subset din dataset public)
- [x] Cod propriu sau clar atribuit (surse citate în Bibliografie)x
---

## Note Finale

**Versiune document:** FINAL pentru examen  
**Ultima actualizare:** [02.02.2026]  
**Tag Git:** `v0.6-optimized-final`

---

*Acest README servește ca documentație principală pentru Livrabilul 1 (Aplicație RN). Pentru Livrabilul 2 (Prezentare PowerPoint), consultați structura din RN_Specificatii_proiect.pdf.*
