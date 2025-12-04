# ReteleNeuronaleProiect
**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Lungeanu Andrei-Alexandru  
**Data:** 27/11/2025

1. Structura Repository-ului Github (versiunea Etapei 4)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          caracteristici_dataset , desciere_set_date , descriere_caracteristici
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Dataset simulat pentru ASAG (Automatic Short Answer Grading), bazat pe concepte tehnice de Retele Neuronale si NLP.
* **Modul de achizitie:** ☑ Generare programatica (Script Python cu variatii controlate ale raspunsurilor).
* **Perioada / conditiile colectarii:** Noiembrie 2025 - Date generate pentru a simula raspunsurile studentilor la 30 de intrebari specifice.

### 2.2 Caracteristicile dataset-ului

* **Numar total de observatii:** 1,500 (30 intrebari x 50 raspunsuri per intrebare).
* **Numar de caracteristici (features):** 6
* **Tipuri de date:** ☑ Numerice / ☑ Categoriale / ☐ Temporale / ☐ Imagini (Textuale).
* **Format fisiere:** ☑ CSV / ☐ TXT / ☐ JSON / ☐ PNG / ☐ Altele: [...]

### 2.3 Descrierea fiecarei caracteristici

| **Caracteristica** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| `question_id` | categorial | – | Identificatorul unic al intrebarii | Q01 – Q30 |
| `question_text` | text | – | Enuntul intrebarii de examen | String |
| `answer_correct` | text | – | Raspunsul de referinta (barem) | String (lungime variabila) |
| `score_range` | numeric | puncte | Punctajul maxim al intrebarii | 1.0 – 5.0 |
| `answer_student` | text | – | Raspunsul simulat al studentului | String |
| `score_manual` | numeric | puncte | Nota acordata (Target Label) | 0.0 – 5.0 |

**Fisier recomandat:** `data/README.md`

---

## 3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Distributia scorurilor:** Analiza distributiei notelor (`score_manual`) a aratat o acoperire a intregului spectru (0-5), asigurand date pentru raspunsuri corecte, partiale si gresite.
* **Lungimea textului:** S-a analizat numarul de cuvinte pentru a filtra raspunsurile prea scurte (sub 2 cuvinte) sau excesiv de lungi.
* **Intrebari:** Dataset-ul contine exact 30 de clase distincte (intrebari unice).

### 3.2 Analiza calitatii datelor

* **Detectarea valorilor lipsa:** 0% valori lipsa (dataset generat controlat). S-a rulat un script de verificare pentru a elimina orice rand cu valori NULL.
* **Consistenta:** S-a verificat ca `score_manual` sa nu depaseasca niciodata `score_range`.

### 3.3 Probleme identificate

* **Variatii textuale:** Textul brut continea majuscule inconsistente si semne de punctuatie care nu sunt relevante pentru analiza semantica.
* **Formatare:** Necesitatea eliminarii spatiilor multiple si a caracterelor speciale.

---

## 4. Preprocesarea Datelor

### 4.1 Curatarea datelor

* **Eliminare valori nule:** Script automat pentru eliminarea randurilor cu NaN in intrebarile sau raspunsurile studentilor.
* **Curatare Text (NLP):**
  * Conversie la litere mici (lowercasing).
  * Eliminarea semnelor de punctuatie (regex).
  * Eliminarea spatiilor albe suplimentare (strip/trim).

### 4.2 Transformarea caracteristicilor

* **Normalizare text:** S-a aplicat o functie `clean_text` pe toate coloanele de tip text (`question_text`, `answer_correct`, `answer_student`).
* **Pregatire Vectorizare:** Textul este pregatit pentru a fi transformat in embedding-uri in etapa urmatoare (folosind BERT/Transformers).

### 4.3 Structurarea seturilor de date

**Impartire realizata pe baza de intrebari (pentru a evita data leakage):**
* **Train (Antrenare):** Intrebarile Q01 – Q24 (1200 inregistrari). Contine perechi complete (intrebare, student, nota).
* **Validation (Validare):** Intrebarile Q25 – Q27 (150 inregistrari). Folosit pentru verificarea generalizarii in timpul antrenarii.
* **Test (Testare):** Intrebarile Q28 – Q30 (150 inregistrari). Date complet noi pentru evaluarea finala a performantei.

### 4.4 Salvarea rezultatelor preprocesarii

* Datele brute validate au fost salvate in `data/raw/`.
* Datele curatate si impartite au fost salvate in folderele `data/train/`, `data/validation/`, `data/test/`.

---

## 5. Fisiere Generate in Aceasta Etapa

* `data/raw/asag_simulated_train_data.csv` – datasetul complet (1500 randuri).
* `data/processed/processed.csv` – setul de date curatat si verificat.
* `data/train/train.csv` – setul de antrenament.
* `data/validation/validation.csv` – setul de validare.
* `data/test/test.csv` – setul de testare.
* `src/preprocessing/process_data.py` – codul Python utilizat pentru curatare si splitare.

---

## 6. Stare Etapa

- [x] Structura repository configurata
- [x] Dataset analizat si generat (1500 intrari)
- [x] Date preprocesate (NLP cleaning)
- [x] Seturi train/val/test generate
- [x] Documentatie actualizata in README

---

## P3

### 1. Tabelul Nevoie Reală → Soluție CPS → Modul Software
Nevoie reala concreta | Cum o rezolva SIA-ul vostru | Modul software responsabil
Reducerea timpului masiv de corectare manuala a testelor scrise (estimat la 30-60 min/test) | Evaluare automata instanta a raspunsurilor textuale -> nota generata in < 5 secunde/raspuns |Neural Network + Scoring Module
Eliminarea subiectivitatii si inconsistentei in notarea raspunsurilor deschise (eroare umana ~15%) | Calcularea scorului de similaritate semantica fata de barem cu o acuratete estimata de > 85% | Preprocessing + Neural Network (Transformer)
Gestionarea volumului mare de studenti si necesitatea feedback-ului rapid | Procesarea simultana a cererilor si stocarea rezultatelor pentru 1000+ studenti fara intarzieri | Web Service + Data Logging


### 2. Diagrama State Machine a Întregului Sistem
IDLE -> WAIT_SUBMISSION (student input) -> RECEIVE_TEXT -> 
VALIDATE_INPUT (not empty, language check) ->
  ├─ [Valid] -> PREPROCESS_TEXT (clean, tokenize) -> RN_INFERENCE (Transformer Embedding) -> 
               CALCULATE_SIMILARITY (Cosine vs Reference) -> MAP_TO_GRADE -> 
               GENERATE_FEEDBACK -> LOG_RESULT -> IDLE (loop)
  └─ [Invalid] -> GENERATE_ERROR_MSG -> LOG_ERROR -> IDLE (loop)
       ↓ [System Update / Maintenance]
     SAFE_SHUTDOWN -> STOP

### Justificarea State Machine-ului ales:

[cite_start]Am ales arhitectura de **procesare secventiala a textului (NLP Pipeline)** pentru ca proiectul nostru vizeaza **reducerea timpului de corectare** [cite: 14] [cite_start]si **eliminarea subiectivitatii**[cite: 6], necesitand un flux liniar si determinist de transformare a textului brut in nota numerica finala.

Starile principale sunt:
1. **[WAIT_SUBMISSION]**: Sistemul asteapta pasiv input-ul de la interfata studentului (stare IDLE cu consum redus de resurse).
2. **[PREPROCESS_TEXT]**: Curatarea automata (lowercasing, eliminare punctuatie, tokenizare) pentru a normaliza datele inainte de intrarea in retea.
3. **[RN_INFERENCE]**: Reteaua Neuronala Transformer proceseaza textul si il transforma intr-un vector semantic dens (embedding).
4. [cite_start]**[CALCULATE_SIMILARITY]**: Algoritmul compara matematic vectorul studentului cu cel al baremului (raspuns corect) si mapeaza distanta la o nota (1-10)[cite: 48].

Tranzitiile critice sunt:
- **[VALIDATE_INPUT]** → **[PREPROCESS_TEXT]**: Se intampla doar cand textul trece verificarile de integritate (nu este gol, este string valid).
- **[ANY_STATE]** → **[LOG_ERROR]**: Se declanseaza cand apar exceptii (ex: text corupt, caractere necodabile, timeout la baza de date).

Starea **LOG_ERROR** este esentiala pentru ca in context educational studentii pot trimite raspunsuri gresite tehnic (fisiere corupte, input gol) sau pot aparea probleme de conexiune la server, iar sistemul trebuie sa inregistreze eroarea fara a se bloca (crash) pentru ceilalti utilizatori.

[cite_start]Bucla de feedback functioneaza astfel: Profesorul valideaza notele generate automat, iar eventualele corectii manuale sunt salvate in baza de date pentru a fi incluse in setul de antrenament (`train.csv`) pentru **auto-invatarea** si rafinarea modelului[cite: 54].


## Checklist Final – Bifati Totul Inainte de Predare

### Documentatie si Structura
- [x] Tabelul Nevoie -> Solutie -> Modul complet 
- [x] Diagrama State Machine creata si salvata si postata alaturi de acest readme pe moodle la P3. State Machine pentru proiectul SAF
- [x] Legenda State Machine scrisa in acest readme 
