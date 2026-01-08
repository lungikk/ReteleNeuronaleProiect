import pandas as pd
import numpy as np
import os
import random

PROJECT_ROOT = r"C:\FACULTATE\ANUL 3 SEM 1\RN"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'asag_simulated_train_data.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f" Voi salva fisierul AICI: {OUTPUT_FILE}")

qa_database = [
    ("Ce este un neuron artificial?", "O unitate de calcul fundamentala intr-o retea neuronala, inspirata biologic."),
    ("Ce reprezinta 'weights' (ponderile)?",
     "Numere care determina importanta input-ului in calculul iesirii neuronului."),
    ("Ce este 'bias' intr-un neuron?", "O valoare adaugata la suma ponderata pentru a decala functia de activare."),
    ("Definiti functia de activare.", "O functie matematica ce introduce non-linearitate in retea."),
    ("Ce face functia Sigmoid?", "Transforma orice valoare de intrare intr-un numar intre 0 si 1."),
    ("Ce este ReLU?", "Rectified Linear Unit, o functie care returneaza x daca x>0, altfel 0."),
    ("Ce este un Layer (strat) ascuns?",
     "Un strat de neuroni intre input si output unde au loc calculele intermediare."),
    ("Definiti Input Layer.", "Primul strat al retelei care primeste datele brute."),
    ("Definiti Output Layer.", "Ultimul strat care produce predictia finala a retelei."),
    ("Ce inseamna Feedforward?", "Procesul de trecere a datelor prin retea de la intrare spre iesire."),
    ("Ce este Loss Function (Functia de Cost)?", "Masoara diferenta dintre predictia retelei si valoarea reala."),
    ("Ce este MSE (Mean Squared Error)?", "O functie de cost care calculeaza media patratelor erorilor."),
    ("Definiti Backpropagation.", "Algoritmul de calculare a gradientului erorii pentru actualizarea ponderilor."),
    ("Ce este Gradient Descent?", "Un algoritm de optimizare folosit pentru a minimiza functia de cost."),
    ("Ce reprezinta Learning Rate?",
     "Un hiperparametru care controleaza cat de mult se modifica ponderile la fiecare pas."),
    ("Ce se intampla daca Learning Rate e prea mare?", "Modelul poate oscila si nu va converge catre solutia optima."),
    ("Ce se intampla daca Learning Rate e prea mic?",
     "Antrenarea va fi foarte lenta si poate ramane blocata in minime locale."),
    ("Ce este o Epoca (Epoch)?", "O trecere completa a intregului set de date prin reteaua neuronala."),
    ("Ce este un Batch?", "Un subset de date procesat simultan inainte de actualizarea ponderilor."),
    ("Definiti Iteratia.", "Numarul de batch-uri necesare pentru a completa o epoca."),
    ("Ce este Overfitting?", "Cand modelul invata zgomotul din datele de antrenare si nu generalizeaza pe date noi."),
    ("Cum prevenim Overfitting-ul?", "Folosind regularizare, dropout sau mai multe date de antrenare."),
    ("Ce este Underfitting?", "Cand modelul este prea simplu pentru a invata structura datelor."),
    ("Ce este Dropout?", "O tehnica de regularizare care dezactiveaza aleatoriu neuroni in timpul antrenarii."),
    ("Ce este un Tensor?", "O structura de date multidimensionala, generalizarea matricelor."),
    ("Diferenta dintre Supervised si Unsupervised Learning?",
     "In Supervised avem etichete (raspunsuri corecte), in Unsupervised nu."),
    ("Ce este Clasificarea?", "O problema unde modelul trebuie sa prezica o categorie discreta."),
    ("Ce este Regresia?", "O problema unde modelul trebuie sa prezica o valoare continua."),
    ("Ce este Matricea de Confuzie?", "Un tabel folosit pentru a evalua performanta unui model de clasificare."),
    ("Definiti Acuratetea (Accuracy).", "Raportul dintre predictiile corecte si numarul total de predictii."),
    ("Ce este Precision?", "Procentul de rezultate pozitive care sunt cu adevarat pozitive."),
    ("Ce este Recall?", "Capacitatea modelului de a gasi toate cazurile pozitive relevante."),
    ("Ce este F1-Score?", "Media armonica intre Precision si Recall."),
    ("Ce este un CNN (Convolutional Neural Network)?",
     "O retea specializata in procesarea datelor tip grila, cum ar fi imaginile."),
    ("Ce face un strat de Convolutie?", "Aplica filtre pentru a extrage trasaturi vizuale din imagine."),
    ("Ce este Pooling?", "O operatie de reducere a dimensiunii hartilor de trasaturi (ex: Max Pooling)."),
    ("Ce este un RNN (Recurrent Neural Network)?",
     "O retea cu memorie, folosita pentru date secventiale (text, timp)."),
    ("Ce este LSTM?", "Long Short-Term Memory, un tip de RNN care rezolva problema memoriei pe termen lung."),
    ("Ce este NLP (Natural Language Processing)?",
     "Domeniul AI care se ocupa cu interactiunea dintre calculatoare si limbajul uman."),
    ("Ce este Tokenizarea?", "Procesul de impartire a textului in unitati mai mici numite tokeni."),
    ("Ce este Word Embedding?", "Reprezentarea cuvintelor ca vectori de numere reale."),
    ("Ce este Transfer Learning?", "Utilizarea unui model pre-antrenat pe un task nou similar."),
    ("Ce este PyTorch?", "O biblioteca open-source de Machine Learning dezvoltata de Facebook."),
    ("Ce este TensorFlow?", "O biblioteca open-source de Machine Learning dezvoltata de Google."),
    ("Ce este Scikit-Learn?", "O biblioteca Python pentru machine learning clasic (nu deep learning)."),
    ("Ce este Pandas?", "O biblioteca Python pentru manipularea si analiza datelor tabulare."),
    ("Ce este NumPy?", "O biblioteca fundamentala pentru calcul stiintific in Python."),
    ("Ce rol are GPU in Deep Learning?",
     "Permite paralelizarea masiva a calculelor matriciale, accelerand antrenarea."),
    ("Ce este Data Augmentation?", "Generarea de date noi prin modificarea usoara a celor existente."),
    ("Care este diferenta dintre parametru si hiperparametru?",
     "Parametrii sunt invatati de model, hiperparametrii sunt setati de programator.")
]

print(f"Generare dataset cu {len(qa_database)} intrebari x 30 raspunsuri = 1500 intrari...")

data = []

prefixes_good = ["Raspunsul este:", "Cred ca este", "Definitia este:", ""]

for q_idx, (q_text, ans_correct) in enumerate(qa_database):
    q_id = f"Q{q_idx + 1:02d}"

    for _ in range(30):
        rand_val = random.random()

        row = {
            'question_id': q_id,
            'question_text': q_text,
            'answer_correct': ans_correct,
            'score_range': 5.0,
            'answer_student': "",
            'score_manual': 0.0
        }

        if rand_val > 0.60:
            prefix = random.choice(prefixes_good)
            student_text = f"{prefix} {ans_correct}".strip()
            if random.random() > 0.5: student_text = student_text.lower()
            row['answer_student'] = student_text
            row['score_manual'] = 5.0

        elif rand_val > 0.40:
            words = ans_correct.split()
            if len(words) > 3:
                student_text = " ".join(words[:-1]) + "."
            else:
                student_text = ans_correct
            row['answer_student'] = student_text
            row['score_manual'] = 4.0

        elif rand_val > 0.20:
            split_point = len(ans_correct) // 2
            row['answer_student'] = ans_correct[:split_point] + "..."
            row['score_manual'] = 2.5

        else:
            wrong_answers = ["Nu stiu.", "Eroare.", "Altceva.", "Nu am invatat.", "Paris."]
            row['answer_student'] = random.choice(wrong_answers)
            row['score_manual'] = 0.0

        data.append(row)

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)
print(f" Verifica acum folderul: {OUTPUT_DIR}")
