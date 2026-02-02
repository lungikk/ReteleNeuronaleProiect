# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Lungeanu Andrei-Alexandru]  
**Data:** [20.11.2025]  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
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
```

---

## 2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** Dataset public (Mohler - University of North Texas) pentru Short Answer Grading, extins prin tehnici de augmentare.
* **Modul de achizitie:** [x] Fisier extern (CSV) / [x] Generare programatica (Data Augmentation)
* **Perioada / conditiile colectarii:** Date istorice (Computer Science dataset) + Augmentare realizata in Decembrie 2024 - Ianuarie 2025.

### 2.2 Caracteristicile dataset-ului

* **Numar total de observatii:** 1500 (dupa augmentare)
* **Numar de caracteristici (features):** 4 coloane principale (input text + target)
* **Tipuri de date:** [ ] Numerice / [ ] Categoriale / [ ] Temporale / [x] Text (NLP)
* **Format fisiere:** [x] CSV / [ ] TXT / [ ] JSON / [ ] PNG / [ ] Altele: [...]

### 2.3 Descrierea fiecarei caracteristici

| Caracteristica | Tip | Unitate | Descriere | Domeniu valori |
|---|---|---|---|---|
| question_id | categorial | - | Identificator unic al intrebarii | String / ID |
| answer_student | text | - | Raspunsul oferit de student (Input) | Text liber |
| answer_correct | text | - | Baremul / Raspunsul de referinta | Text liber |
| score_manual | numeric | puncte | Nota acordata de profesor (Target) | 0.0 - 5.0 |

**Fisier recomandat:** `data/README.md`

---

## 3. Analiza Exploratorie a Datelor (EDA) - Sintetic

### 3.1 Statistici descriptive aplicate

* **Distributia notelor:** Analiza frecventei notelor (0.0, 2.5, 4.0, 5.0) a aratat initial un dezechilibru (multe note de 5.0, putine intermediare).
* **Lungimea textului:** Calculul numarului mediu de cuvinte per raspuns pentru a detecta raspunsurile prea scurte ("sparse features").
* **Vocabular:** Identificarea celor mai frecvente cuvinte (fara stop-words) pentru a calibra vectorizatorul TF-IDF.

### 3.2 Analiza calitatii datelor

* **Detectarea valorilor lipsa:** Verificare integritate CSV (nu s-au gasit valori NULL in coloanele de text).
* **Detectarea duplicatelor:** Eliminarea raspunsurilor identice inainte de split-ul train/test pentru a evita "data leakage".
* **Verificare consistenta:** Identificarea raspunsurilor care contineau doar semne de punctuatie sau caractere irelevante.

### 3.3 Probleme identificate

* **Dezechilibru de clase:** Clasele extreme (0.0 si 5.0) erau majoritare. S-a rezolvat prin augmentarea claselor intermediare (4.0).
* **Variabilitate lexicala:** Studentii folosesc sinonime care nu apar in barem (ex: "eroare" vs "greseala"), ceea ce necesita o vectorizare robusta sau augmentare.
* **Raspunsuri foarte scurte:** Texte de 1-2 cuvinte care nu ofera suficient context pentru TF-IDF.

---

## 4. Preprocesarea Datelor

### 4.1 Curatarea datelor

* **Eliminare duplicate:** S-au sters randurile identice.
* **Normalizare text:**
  * Lowercasing (conversie la litere mici).
  * Eliminare punctuatie si caractere speciale.
  * (Optional) Eliminare stop-words (cuvinte de legatura).
* **Tratarea valorilor lipsa:** Nu a fost cazul, dataset-ul fiind curat.

### 4.2 Transformarea caracteristicilor

* **Vectorizare (Feature Engineering):**
  * Aplicare **TF-IDF (Term Frequency - Inverse Document Frequency)**.
  * Max features: 1000 (cele mai relevante cuvinte).
  * Input combinat: `answer_student` + `answer_correct` concatinate pentru a oferi context retelei.
* **Encoding Target:** Valorile float (0.0 - 5.0) au fost pastrate ca atare pentru regresie (MLPRegressor).

### 4.3 Structurarea seturilor de date

**Impartire realizata:**
* 70% - train (Antrenare model)
* 15% - validation (Tuning hiperparametri)
* 15% - test (Evaluare finala performanta)

**Principii respectate:**
* **Stratificare:** S-a folosit `stratify=y` pentru a pastra aceeasi distributie a notelor in toate cele 3 seturi.
* **Fara scurgere de informatie:** Vectorizatorul TF-IDF a fost antrenat (`fit`) DOAR pe setul de train, apoi aplicat (`transform`) pe validation si test.

### 4.4 Salvarea rezultatelor preprocesarii

* Date preprocesate in `data/processed/`.
* Seturi finale in `data/train/`, `data/validation/`, `data/test/`.
* Vectorizatorul salvat in `config/preprocessing_params.pkl` (sau `models/vectorizer.pkl`).

---

## 5. Fisiere Generate in Aceasta Etapa

* `data/raw/` - datele brute (Mohler original).
* `data/generated/` - datele augmentate (parafrazari).
* `data/train/`, `data/validation/`, `data/test/` - fisiere CSV gata de antrenare.
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

## 6. Stare Etapa

- [x] Structura repository configurata
- [x] Dataset analizat (EDA realizata)
- [x] Date preprocesate (Curatare + TF-IDF)
- [x] Seturi train/val/test generate si stratificate
- [x] Documentatie actualizata in README + `data/README.md`

---