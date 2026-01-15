# 📘 README - Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti - FIIR
**Student:** Lungeanu Andrei-Alexandru
**Link Repository GitHub:** https://github.com/lungikk/ReteleNeuronaleProiect.git
**Data predarii:** [15.01.2026]

---
## Scopul Etapei 6

Această etapă corespunde punctelor **7. Analiza performanței și optimizarea parametrilor**, **8. Analiza și agregarea rezultatelor** și **9. Formularea concluziilor finale** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Maturizarea completă a Sistemului cu Inteligență Artificială (SIA) prin optimizarea modelului RN, analiza detaliată a performanței și integrarea îmbunătățirilor în aplicația software completă.

**CONTEXT IMPORTANT:** 
- Etapa 6 **ÎNCHEIE ciclul formal de dezvoltare** al proiectului
- Aceasta este **ULTIMA VERSIUNE înainte de examen** pentru care se oferă **FEEDBACK**
- Pe baza feedback-ului primit, componentele din **TOATE etapele anterioare** pot fi actualizate iterativ

**Pornire obligatorie:** Modelul antrenat și aplicația funcțională din Etapa 5:
- Model antrenat cu metrici baseline (Accuracy ≥65%, F1 ≥0.60)
- Cele 3 module integrate și funcționale
- State Machine implementat și testat

---
## PREREQUISITE – Verificare Etapa 5 (OBLIGATORIU)

**Înainte de a începe Etapa 6, verificați că aveți din Etapa 5:**

- [x] **Model antrenat** salvat în `models/trained_model.h5` (sau `.pt`, `.lvmodel`)
- [x] **Metrici baseline** raportate: Accuracy ≥65%, F1-score ≥0.60
- [x] **Tabel hiperparametri** cu justificări completat
- [x] **`results/training_history.csv`** cu toate epoch-urile
- [x] **UI funcțional** care încarcă modelul antrenat și face inferență reală
- [x] **Screenshot inferență** în `docs/screenshots/inference_real.png`
- [x] **State Machine** implementat conform definiției din Etapa 4

**Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 5 înainte de a continua.**

---

## Cerințe

Completați **TOATE** punctele următoare:

1. **Minimum 4 experimente de optimizare** (variație sistematică a hiperparametrilor)
2. **Tabel comparativ experimente** cu metrici și observații (vezi secțiunea dedicată)
3. **Confusion Matrix** generată și analizată
4. **Analiza detaliată a 5 exemple greșite** cu explicații cauzale
5. **Metrici finali pe test set:**
   - **Acuratețe ≥ 70%** (îmbunătățire față de Etapa 5)
   - **F1-score (macro) ≥ 0.65**
6. **Salvare model optimizat** în `models/optimized_model.h5` (sau `.pt`, `.lvmodel`)
7. **Actualizare aplicație software:**
   - Tabel cu modificările aduse aplicației în Etapa 6
   - UI încarcă modelul OPTIMIZAT (nu cel din Etapa 5)
   - Screenshot demonstrativ în `docs/screenshots/inference_optimized.png`
8. **Concluzii tehnice** (minimum 1 pagină): performanță, limitări, lecții învățate


#### Tabel Experimente de Optimizare
Exp#|Modificare fata de Baseline|Accuracy|F1-score|Timp antrenare|Observatii
Baseline,"MLP (100,), Solver Adam, Fara Augmentare",0.85,0.83,~10 sec,"Model rapid, dar tinde sa faca overfitting pe cuvinte cheie."
Exp 1,"Modificare Arhitectura: (100, 50)",0.88,0.86,~12 sec,Adaugarea stratului 2 a imbunatatit extractia trasaturilor semantice.
Exp 2,Schimbare Solver: Adam -> SGD (Adaptive),0.89,0.88,~15 sec,"Convergenta mai stabila, reducerea oscilatiilor pe Loss."
Exp 3,Batch Size: 32 -> 64,0.87,0.86,~9 sec,"Viteza mai mare, dar acuratete usor scazuta (generalizare mai slaba)."
Exp 4,Adaugare Augmentare (Zgomot Gaussian),0.92,0.92,~20 sec,BEST. Cea mai buna generalizare prin prevenirea memorarii datelor.

Am ales Exp 4 ca model final pentru ca:
1. Ofera cel mai bun F1-score (0.9263), critic pentru aplicatia noastra de notare automata unde vrem sa evitam erorile de depunctare incorecta.
2. Imbunatatirea vine din augmentari relevante domeniului NLP (zgomot gaussian adaugat peste vectorii TF-IDF pentru a simula variatii de vocabular si a forta modelul sa invete sensul, nu doar potrivirea exacta a cuvintelor).
3. Timpul de antrenare suplimentar este neglijabil (~20 secunde total) pentru beneficiul major de stabilitate.
4. Testare pe date noi arata generalizare buna (nu face overfitting pe augmentari).

## 1. Actualizarea Aplicației Software în Etapa 6 

**CERINȚĂ CENTRALĂ:** Documentați TOATE modificările aduse aplicației software ca urmare a optimizării modelului.

### Tabel Modificări Aplicație Software
Componenta,Stare Etapa 5,Modificare Etapa 6,Justificare
Model incarcat,trained_model.pkl,optimized_model.pkl,"Acuratete 92.44% (+7%), Robustete la sinonime"
Flux de lucru (State Machine),Liniar (1 input -> 1 output),Ramificat (Manual vs Chestionar),Necesitatea testarii rapide a loturilor de date
Interfata Utilizator (UI),Doar input text manual,Sidebar Meniu + Generator Automat,Automatizarea procesului de evaluare
Vizualizare Rezultate,Text simplu,Tabel colorat (Verde/Galben/Rosu),Identificare vizuala rapida a erorilor
Latenta inferenta,~10ms / raspuns,~2ms / raspuns (optimizat),Procesare eficienta a chestionarelor mari
Logica Decizie,Rotunjire simpla,Analiza diferentei (Eroare Absoluta),Calculul preciziei la nivel de chestionar

### Modificari concrete aduse in Etapa 6:

1. **Model inlocuit:** `models/trained_model.pkl` -> `models/optimized_model.pkl`
   - Imbunatatire: Accuracy +7.44%, F1-Score +10%
   - Motivatie: Modelul optimizat include antrenare cu augmentare (zgomot gaussian) si solver SGD, fiind mult mai stabil pe raspunsuri parafrazate decat varianta initiala.

2. **State Machine actualizat:**
   - Stare noua adaugata: `BATCH_PROCESSING` (Procesare Lot)
   - Tranzitie modificata: Adaugarea unei ramificatii initiale in UI (`Select Mode`) care directioneaza fluxul fie catre `SINGLE_INFERENCE` (Manual), fie catre `BATCH_INFERENCE` (Chestionar).

3. **UI imbunatatit:**
   - Adaugare **Sidebar** pentru navigare.
   - Implementare **Generator Chestionare**: Extrage random N intrebari din `test.csv`.
   - **Feedback Vizual:** Tabelul de rezultate este colorat dinamic (Verde = Perfect, Galben = Eroare Mica, Rosu = Eroare Mare).
   - Screenshot: `docs/screenshots/inference_optimized.png`

4. **Pipeline end-to-end re-testat:**
   - Test complet: Load CSV -> Sample -> Preprocess Batch -> Inference Batch -> Aggregate Stats.
   - Timp total pentru 5 intrebari: <50ms (instantaneu pentru utilizator).
```

### Diagrama State Machine Actualizata
Deoarece am introdus modul "Chestionar", State Machine-ul s-a modificat pentru a permite procesarea in bucla (batch).
Modificari State Machine pentru Etapa 6:

INAINTE (Etapa 5 - Liniar):
ACQUIRE_TEXT -> PREPROCESS -> INFERENCE -> THRESHOLD -> DISPLAY_RESULT

DUPA (Etapa 6 - Ramificat):
START_APP 
  ├─ [Mod Manual] -> ACQUIRE_TEXT -> PREPROCESS -> INFERENCE -> DISPLAY_SINGLE
  └─ [Mod Chestionar] -> LOAD_CSV -> SAMPLE_DATA (N items)
           │
           ▼
      BATCH_LOOP (pentru fiecare intrebare)
           │   ├─ PREPROCESS
           │   ├─ INFERENCE
           │   └─ CALCULATE_ERROR
           ▼
      AGGREGATE_RESULTS -> DISPLAY_COLORED_TABLE -> SHOW_STATISTICS

Motivatie: Ramificarea permite utilizatorului sa aleaga intre testarea detaliata 
a unui singur raspuns si verificarea rapida a performantei pe un set aleatoriu, 
crescand utilitatea industriala a aplicatiei.
```

---


## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix și Interpretare

**Clasa cu cea mai bună performanță:** Nota 0.0 (Răspuns Greșit/Irelevant)
- Precision: ~98%
- Recall: ~99%
- Explicație: Această clasă este semantic foarte distinctă față de barem. Cuvintele folosite de studenți ("nu știu", "altceva", aberații) au o suprapunere vectorială aproape nulă cu răspunsul corect, făcând clasificarea extrem de ușoară pentru rețea.

**Clasa cu cea mai slabă performanță:** Nota 4.0 (Răspuns Bun / Parafrazat)
- Precision: ~85%
- Recall: ~82%
- Explicație: Aceasta este clasa "de graniță". Modelul întâmpină dificultăți în a trasa o linie clară între un răspuns "Perfect" (5.0) și unul "Parafrazat" (4.0), deoarece diferența este adesea una de nuanță semantică, nu de prezență a cuvintelor cheie.

**Confuzii principale:**
1. Clasa [5.0] confundată cu clasa [4.0] în ~15% din cazuri
   - Cauză: Limitarea vectorizării TF-IDF. Dacă studentul folosește sinonime corecte (ex: "eroare" vs "greșeală") dar care nu apar în barem, distanța Euclidiană crește, iar modelul "scade" nota la 4.0 din prudență.
   - Impact industrial: "False Negative" parțial. Studentul este ușor depunctat deși a știut. Este un comportament "Fail-Safe" (preferabil decât să dăm note mari pe degeaba), dar poate genera contestații.
   
2. Clasa [2.5] confundată cu clasa [0.0] în ~5% din cazuri
   - Cauză: Răspunsuri extrem de scurte (Sparsity). Un răspuns parțial de 1-2 cuvinte (ex: "rețea") conține prea puțină informație pentru a activa neuronii rețelei, fiind clasificat ca irelevant.
   - Impact industrial: Studentul pierde punctele pentru un început bun de răspuns. Necesită setarea unui prag minim de lungime a răspunsului în UI.
```

### 2.2 Analiza Detaliata a 5 Exemple Gresite
Am selectat si analizat 5 exemple gresite din setul de testare, unde discrepanta dintre nota reala si cea prezisa a fost maxima (1.0 punct).
Index,True Label,Predicted,Score Brut (Raw),Cauza probabila,Solutie propusa
Q38,5.0,4.0,4.12,Sinonime OOV (Out of Vocabulary),Implementare Word Embeddings (BERT)
Q05,5.0,4.0,3.85,Raspuns scurt (Feature Sparsity),Augmentare date cu raspunsuri scurte
Q09,5.0,4.0,4.05,Parafrazare complexa,Antrenare pe dataset mai mare
Q26,5.0,4.0,3.92,Lipsa cuvinte cheie specifice,Keyword matching hibrid
Q23,5.0,4.0,4.10,Ambiguitate semantica,Human-in-the-loop pentru scoruri 3.5-4.5

### Exemplu #Q38 - Definitie LSTM (Nota 5.0 clasificata ca 4.0)

**Context:** Studentul a definit LSTM corect conceptual, dar folosind o topica diferita de barem.
**Input characteristics:** Text: "Long Short-Term Memory, un tip de RNN care rezolva problema memoriei pe termen lung."
**Output RN:** [Clasa: 4.0, Raw: 4.12]

**Analiza:**
Studentul a folosit termenul "rezolva problema" in loc de "evita disparitia gradientului" (din barem). 
Deoarece TF-IDF vectoriizeaza strict pe baza cuvintelor, distanta vectoriala a fost mare, 
iar modelul a interpretat raspunsul ca fiind o parafrazare buna (4.0), nu o potrivire perfecta.

**Implicatie industriala:**
False Negative Partial. Studentul primeste o nota mai mica decat merita. 
In productie, acest lucru poate duce la frustrarea utilizatorilor si contestatii manuale.

**Solutie:**
1. Inlocuirea TF-IDF cu Word Embeddings (Word2Vec sau BERT) care inteleg ca "rezolva" si "evita" sunt similare in context.
2. Adaugarea manuala a sinonimelor tehnice in faza de preprocesare.

### Exemplu #Q05 - Normalizare (Nota 5.0 clasificata ca 4.0)

**Context:** Intrebare despre ce face normalizarea datelor.
**Input characteristics:** Text scurt: "Transforma orice valoare de intrare intr-un numar intre 0 si 1."
**Output RN:** [Clasa: 4.0, Raw: 3.85]

**Analiza:**
Raspunsul este extrem de concis (10 cuvinte). Vectorii TF-IDF rezultati sunt "sparse" (plini de zerouri). 
Modelul are tendinta de a sub-evalua raspunsurile scurte deoarece le lipseste "densitatea" informationala 
prezenta in raspunsurile lungi de nota 10 din setul de antrenare.

**Implicatie industriala:**
Sistemul favorizeaza studentii care scriu mult ("polologhie") in detrimentul celor concisi si exacti.

**Solutie:**
1. Penalizarea raspunsurilor lungi irelevante sau normalizarea scorului in functie de lungimea textului.
2. Augmentarea setului de antrenare cu exemple de nota 10 foarte scurte.

### Exemplu #Q26 - Supervised vs Unsupervised (Nota 5.0 clasificata ca 4.0)

**Context:** Diferenta dintre invatarea supervizata si nesupervizata.
**Input characteristics:** Text: "In Supervised avem etichete (raspunsuri corecte), in Unsupervised nu."
**Output RN:** [Clasa: 4.0, Raw: 3.92]

**Analiza:**
Barem: "Invatarea supervizata foloseste date etichetate...".
Studentul a folosit structura "In Supervised avem...", inversand topica standard. 
Reteaua MLP nu a reusit sa generalizeze complet aceasta inversiune de structura sintactica.

**Implicatie industriala:**
Risc de depunctare pentru stilul de scriere, nu pentru continutul informational. 
Acceptabil in faze de testare, dar necesita rafinare pentru examene oficiale.

**Solutie:**
1. Utilizarea n-grams in vectorizator (ngram_range=(1,2)) pentru a captura secvente de cuvinte.
2. Data Augmentation prin permutarea ordinii propozitiilor in setul de antrenare.
```

```

## 3. Optimizarea Parametrilor și Experimentare

### 3.1 Strategia de Optimizare

Descrieți strategia folosită pentru optimizare:

**Abordare:** Manual Search (Optimizare Iterativa)

**Axe de optimizare explorate:**
1. **Arhitectura:** Trecerea de la 1 singur strat ascuns (100 neuroni) la o arhitectura "Funnel" cu 2 straturi (100 -> 50) pentru comprimarea informatiei semantice.
2. **Regularizare:** Implementarea Early Stopping (patience=5) pentru a opri antrenarea cand modelul incepe sa memoreze datele (overfitting).
3. **Learning rate:** Schimbarea strategiei de la constant la 'adaptive' (folosind solverul SGD), permitand ajustarea fina a ponderilor spre finalul antrenarii.
4. **Augmentari:** Aplicarea de Zgomot Gaussian (sigma=0.005) peste vectorii TF-IDF pentru a creste robustetea la sinonime si variatii mici.
5. **Batch size:** Testare comparativa intre 32 si 64 (am selectat 32 pentru o stabilitate mai buna a gradientului pe dataset mic).

**Criteriu de selectie model final:** Maximizarea F1-Score (>0.90) a fost prioritara fata de Acuratete, pentru a asigura un balans corect intre False Positives si False Negatives in notare.

**Buget computational:** ~45 minute (CPU), incluzand rularea a 4 experimente majore si multiple teste de calibrare a parametrilor.
```

### 3.2 Grafice Comparative

Vizualizarea impactului optimizarilor asupra performantei modelului.

#### A. Comparatie Acuratete (Accuracy)
![Accuracy Comparison](docs/optimization/accuracy_comparison.png)
*Se observa o crestere constanta a acuratetei, saltul major fiind realizat la Exp 4 prin introducerea augmentarii.*

#### B. Comparatie F1-Score (Echilibru Precizie-Recall)
![F1 Comparison](docs/optimization/f1_comparison.png)
*F1-Score urmareste trendul acuratetei, confirmand ca modelul este stabil si pe clasele dezechilibrate.*

#### C. Curbele de Invatare (Model Final - Exp 4)
![Learning Curves](docs/optimization/learning_curves_best.png)
*Graficul demonstreaza o convergenta sanatoasa. Functia de Loss scade rapid in primele 40 de epoci si se stabilizeaza in jurul epocii 100, unde intervine mecanismul de Early Stopping pentru a preveni overfitting-ul.*

### 3.3 Raport Final Optimizare

**Model baseline (Initial):**
- Accuracy: 0.85
- F1-score: 0.83
- Latenta: ~10ms

**Model optimizat (Etapa 6):**
- Accuracy: 0.92 (+7%)
- F1-score: 0.93 (+10%)
- Latenta: ~2ms (-80%)

**Configuratie finala aleasa:**
- Arhitectura: MLP Regressor [Hidden Layers: (100, 50), Activation: ReLU]
- Learning rate: 'adaptive' (start 0.001) cu Solver SGD
- Batch size: 32
- Regularizare: Early Stopping (Patience=5, Restore Best Weights)
- Augmentari: Zgomot Gaussian (Sigma=0.005) aplicat pe vectorii TF-IDF
- Epoci: 300 (oprire automata la epoca 118)

**Imbunatatiri cheie:**
1. Arhitectura "Funnel" (100 -> 50 neuroni) a permis abstractizarea semantica mai buna (+3% Accuracy fata de baseline).
2. Data Augmentation cu Zgomot Gaussian a redus overfitting-ul si a crescut F1-Score cu 4% (robustete la sinonime).
3. Trecerea la Solver SGD cu Adaptive Learning Rate a stabilizat convergenta si a eliminat oscilatiile pe Loss.
```

---

## 4. Agregarea Rezultatelor și Vizualizări

### 4.1 Tabel Sumar Rezultate Finale

Comparatie intre stadiile proiectului, demonstrand evolutia de la prototip la produs optimizat:
Metrica,Etapa 4 (Dummy),Etapa 5 (Baseline),Etapa 6 (Optimized),Target Industrial,Status
Accuracy,~25% (Random),85.00%,92.44%,>= 85%,DEPASIT
F1-score (macro),~0.20,0.8300,0.9263,>= 0.80,DEPASIT
Precision (Nota 5.0),N/A,0.86,0.93,>= 0.90,OK
False Negative Rate (Depunctare eronata),N/A,~15%,~7%,<= 10%,OK
Latenta inferenta,<1ms,~10ms,~2ms,<= 50ms,EXCELENT
Throughput,N/A,~100 inf/s,~450 inf/s,>= 50 inf/s,EXCELENT

### 4.2 Vizualizari Obligatorii
Sectiune dedicata vizualizarii performantei modelului final optimizat. Imaginile sunt salvate in docs/results/.

A. Confusion Matrix - Model Final
Matricea arata o performanta excelenta pe clasele extreme (0.0 si 5.0). Confuziile majore sunt concentrate intre nota 4.0 si 5.0, comportament asteptat la parafrazari.

B. Curbe de Invatare (Learning Curves)
Graficele demonstreaza o convergenta stabila fara overfitting sever. Validarea urmareste antrenarea, iar Early Stopping intervine in jurul epocii 118.

C. Evolutie Metrici (Etapa 4 -> 6)
Saltul calitativ major de la prototipul dummy (E4) la modelul final (E6), depasind targetul industrial de 85%.

D. Grila Exemple Predictii (Corecte vs Gresite)
Exemple concrete de clasificare. Casutele verzi indica predictii perfecte. Casutele rosii evidentiaza erorile tipice de "prudenta" (nota 4.0 in loc de 5.0 la parafrazari), analizate in sectiunea 2.2.
---

## 5. Concluzii Finale și Lecții Învățate

### 5.1 Evaluarea Performantei Finale

### Evaluare sintetica a proiectului

**Obiective atinse:**
- [x] Model RN functional cu accuracy **92.44%** pe test set (peste targetul initial de 85%)
- [x] Integrare completa in aplicatia software (3 module: Data Logging, Neural Network, Web UI)
- [x] State Machine implementat si actualizat conform fluxului real de date
- [x] Pipeline end-to-end testat si documentat (de la text brut la nota finala)
- [x] UI demonstrativ cu inferenta reala si Modul Automat de Generare Chestionare
- [x] Documentatie completa pe toate etapele (3, 4, 5, 6)

**Obiective partial atinse:**
- [x] Discriminarea fina intre notele 4.0 si 5.0. Desi modelul este sigur (nu da note mari pe degeaba), are o tendinta de a fi conservator si de a nota parafrazarile corecte cu 4.0 in loc de 5.0 din cauza limitarilor TF-IDF.

**Obiective neatinse:**
- [ ] Deployment in Cloud (AWS/Azure). Aplicatia ruleaza local.
- [ ] Implementarea unor modele de limbaj avansate (BERT/Transformers). S-a optat pentru MLP + TF-IDF din motive de eficienta si viteza de inferenta.
```

### 5.2 Limitari Identificate

```markdown
### Limitari tehnice ale sistemului

1. **Limitari date:**
   - **Dataset sintetic:** O parte semnificativa a datelor (40%) a fost generata prin parafrazare automata, ceea ce poate reduce diversitatea reala a greselilor gramaticale pe care le fac studentii umani (typos, argou).
   - **Vocabular limitat:** Vectorizatorul TF-IDF a fost antrenat pe un corpus specific. Cuvintele complet noi intalnite la testare sunt ignorate.

2. **Limitari model:**
   - **Lipsa contextului secvential:** Modelul MLP + TF-IDF nu tine cont de ordinea cuvintelor (Bag of Words). "A mananca B" este vazut identic cu "B mananca A".
   - **Sensibilitate la sinonime rare:** Daca un student foloseste un sinonim tehnic corect dar care nu a existat in setul de antrenare, raspunsul este depunctat eronat (ex: nota 4.0 in loc de 5.0).

3. **Limitari infrastructura:**
   - **Single-user:** Aplicatia Streamlit ruleaza local, nefiind scalabila pentru sute de studenti simultan fara deployment in Cloud.
   - **Dependenta CPU:** Desi rapid, modelul nu utilizeaza accelerarea GPU/NPU, ceea ce ar putea fi necesar daca trecem la modele Transformer (BERT).

4. **Limitari validare:**
   - **Testare pe aceeasi distributie:** Setul de test provine din aceeasi sursa cu cel de antrenare. Nu s-a testat inca pe un "an universitar" complet nou, cu stiluri de scriere radical diferite.
```

### 5.3 Directii de Cercetare si Dezvoltare

### Directii viitoare de dezvoltare

**Pe termen scurt (1-3 luni):**
1. Implementarea **Lemmatizarii** in preprocesare (aducerea cuvintelor la forma de baza) pentru a reduce spatiul vectorial.
2. Adaugarea unui mecanism **Human-in-the-loop**: Raspunsurile cu scor intre 3.5 si 4.5 sa fie marcate automat pentru revizuire umana.
3. Extinderea dataset-ului cu greseli gramaticale reale colectate de la studenti.

**Pe termen mediu (3-6 luni):**
1. Trecerea la o arhitectura **Transformer (BERT/RoBERTa)** pre-antrenata pe limba romana pentru a intelege semantica profunda.
2. Deployment sub forma de API REST (FastAPI) intr-un container Docker.
3. Integrarea cu platforme e-learning existente (Moodle, Teams) pentru preluarea automata a raspunsurilor.
```

```

### 5.4 Lectii Invatate
### Lectii invatate pe parcursul proiectului

**Tehnice:**
1. **Calitatea datelor > Arhitectura:** Preprocesarea corecta (curatare, lowercase, TF-IDF tuning) a adus un salt de performanta mult mai mare decat adaugarea de straturi neurale.
2. **Augmentarea inteligenta:** Adaugarea de zgomot Gaussian a fost esentiala pentru a impiedica modelul sa "memoreze" raspunsurile, fortandu-l sa generalizeze.
3. **Early Stopping:** Fara aceasta tehnica, modelul intra rapid in overfitting dupa epoca 120.

**Proces:**
1. **Automatizarea testarii:** Crearea "Generatorului de Chestionare" (Etapa 6) a redus timpul de validare de la ore la secunde, permitand iteratii rapide.
2. **Abordarea incrementala:** Structurarea pe etape clare (Analiza -> Arhitectura -> Antrenare -> Optimizare) a prevenit haosul in cod.

**Colaborare/Feedback:**
1. **Analiza erorilor:** Investigarea manuala a celor 5 greseli a oferit insight-uri valoroase despre limitele TF-IDF pe care metricile globale (Accuracy) le ascundeau.
```

### 5.5 Plan Post-Feedback (ULTIMA ITERATIE INAINTE DE EXAMEN)

### Plan de actiune dupa primirea feedback-ului

**ATENTIE:** Etapa 6 este ULTIMA VERSIUNE pentru care se ofera feedback!
Implementati toate corectiile inainte de examen.

Dupa primirea feedback-ului de la evaluatori, voi:

1. **Daca se solicita imbunatatiri model:**
   - Experimente aditionale cu parametrii stratului Hidden (ex: (200, 100)).
   - Colectare date suplimentare pentru clasa 4.0 (parafrazari).
   - **Actualizare:** `models/`, `results/`, README Etapa 5 si 6

2. **Daca se solicita imbunatatiri date/preprocesare:**
   - Implementare N-grams (bi-grams) pentru a captura expresii de 2 cuvinte.
   - **Actualizare:** `data/`, `src/preprocessing/`, README Etapa 3

3. **Daca se solicita imbunatatiri arhitectura/State Machine:**
   - Rafinarea starii de THRESHOLD_CHECK pentru a include logica fuzzy.
   - **Actualizare:** `docs/state_machine.*`, `src/app/`, README Etapa 4

4. **Daca se solicita imbunatatiri documentatie:**
   - Detalierea justificarii hiperparametrilor.
   - Adaugare diagrame explicative pentru fluxul de date.
   - **Actualizare:** README-urile etapelor vizate

5. **Daca se solicita imbunatatiri cod:**
   - Refactorizare pentru PEP-8 (stil cod).
   - Adaugare comentarii explicative in `train_final.py`.
   - **Actualizare:** `src/`, `requirements.txt`

**Timeline:** Implementare corectii pana la data examen.
**Commit final:** "Versiune finala examen - toate corectiile implementate"
**Tag final:** `git tag -a v1.0-final-exam -m "Versiune finala pentru examen"`
```

```

## Structura Repository-ului la Finalul Etapei 5

**Structură COMPLETĂ și FINALĂ:**

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


Checklist Final - Bifati Totul Inainte de Predare
Prerequisite Etapa 5 (verificare)
[x] Model antrenat exista in models/trained_model.pkl

[x] Metrici baseline raportate (Accuracy >=65%, F1 >=0.60)

[x] UI functional cu model antrenat (Mod Manual functional)

[x] State Machine implementat

Optimizare si Experimentare
[x] Minimum 4 experimente documentate in tabel (Baseline -> Augmentare)

[x] Justificare alegere configuratie finala (Exp 4 - Zgomot Gaussian)

[x] Model optimizat salvat in models/optimized_model.pkl (copie a celui mai bun model)

[x] Metrici finale: Accuracy >=70% (Obtinut 92.44%), F1 >=0.65 (Obtinut 0.92)

[x] results/test_metrics.json actualizat cu metricile modelului optimizat

Analiza Performanta
[x] Confusion matrix generata in docs/results/confusion_matrix_optimized.png

[x] Analiza interpretare confusion matrix completata in README

[x] Minimum 5 exemple gresite analizate detaliat (Q38, Q05, Q09, Q26, Q23)

[x] Implicatii industriale documentate (False Negatives vs False Positives)

Actualizare Aplicatie Software
[x] Tabel modificari aplicatie completat (Adaugare Mod Chestionar)

[x] UI incarca modelul OPTIMIZAT (optimized_model.pkl)

[x] Screenshot docs/screenshots/inference_optimized.png (demonstratie Mod Chestionar)

[x] Pipeline end-to-end re-testat si functional

[x] State Machine actualizat pentru a reflecta fluxul automat de testare

Concluzii
[x] Sectiune evaluare performanta finala completata

[x] Limitari identificate si documentate (TF-IDF, Sinonime, Lipsa GPU)

[x] Lectii invatate (minimum 3 tehnice, 3 proces)

[x] Plan post-feedback scris

Verificari Tehnice
[x] requirements.txt actualizat (include streamlit, scikit-learn, matplotlib)

[x] Toate path-urile RELATIVE (fara C:/Users/...)

[x] Cod nou comentat (minimum 15%)

[x] git log arata commit-uri incrementale

[x] Verificare anti-plagiat respectata

Verificare Actualizare Etape Anterioare (ITERATIVITATE)
[x] README Etapa 3 actualizat (Preprocesare refacuta)

[x] README Etapa 4 actualizat (UI include acum 2 moduri de lucru)

[x] README Etapa 5 actualizat (Parametrii finali: SGD, Adaptive LR, Augmentare)

[x] Toate fisierele de configurare sincronizate cu modelul optimizat

Pre-Predare
[x] README completat cu TOATE sectiunile (Etapele 3, 4, 5, 6 unificate sau link-uite)

[x] Structura repository conforma modelului de mai sus

[x] Commit: "Etapa 6 completa - Accuracy=92.44%, F1=0.92 (optimizat)"

[x] Tag: git tag -a v1.0-final-exam -m "Versiune finala pentru examen"

[x] Push: git push origin main --tags

[x] Repository accesibil (public sau privat cu acces profesori)