import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

st.set_page_config(page_title="Sistem Notare Automata", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test', 'test.csv')


@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECTORIZER_PATH)

    if os.path.exists(TEST_DATA_PATH):
        df_test = pd.read_csv(TEST_DATA_PATH)
    else:
        df_test = None

    return model, vect, df_test


model, vectorizer, df_test = load_resources()


def get_nearest_grade(prediction):
    possible_grades = np.array([0.0, 2.5, 4.0, 5.0])
    idx = (np.abs(possible_grades - prediction)).argmin()
    return possible_grades[idx]


def predict_single(student_text, correct_text):
    text_input = str(student_text) + " " + str(correct_text)
    vector_input = vectorizer.transform([text_input])
    vector_dense = vector_input.toarray()
    vector_final = np.array(vector_dense, dtype=np.float32)
    pred_raw = model.predict(vector_final)[0]
    pred_class = get_nearest_grade(pred_raw)
    return pred_class, pred_raw


st.title("🎓 Sistem AI de Notare Automata")

if model is None:
    st.error(" EROARE CRITICA: Nu gasesc modelele (.pkl). Ruleaza train_final.py!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Meniu Principal")
app_mode = st.sidebar.radio("Alege Modul de Lucru:", ["Mod Manual (Testare 1 la 1)", "Mod Chestionar (Testare Random)"])

st.sidebar.markdown("---")
st.sidebar.info("Modul 'Chestionar' extrage intrebari aleatorii din setul de testare si le noteaza automat.")

# ==========================================
# MODUL 1: MANUAL
# ==========================================
if app_mode == "Mod Manual (Testare 1 la 1)":
    st.subheader("📝 Testare Manuala")
    st.markdown("Introdu manual un raspuns pentru a verifica nota.")

    col1, col2 = st.columns(2)
    with col1:
        correct_answer = st.text_area("Barem (Raspuns Corect)",
                                      "Reteaua neuronala este un model inspirat din creierul uman.", height=150)
    with col2:
        student_answer = st.text_area("Raspuns Student", "Este un model matematic inspirat biologic.", height=150)

    if st.button("Calculeaza Nota", type="primary"):
        if not student_answer or not correct_answer:
            st.warning("Completeaza ambele campuri!")
        else:
            grade, raw = predict_single(student_answer, correct_answer)

            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Nota AI", f"{grade}")
            with c2:
                if grade == 5.0:
                    st.success("Excelent! (5.0)")
                elif grade >= 4.0:
                    st.info("Raspuns Bun / Parafrazat (4.0)")
                elif grade >= 2.5:
                    st.warning("Raspuns Partial (2.5)")
                else:
                    st.error("Raspuns Gresit (0.0)")
            st.caption(f"Scor brut: {raw:.4f}")

# ==========================================
# MODUL 2: CHESTIONAR AUTOMAT
# ==========================================
elif app_mode == "Mod Chestionar (Testare Random)":
    st.subheader("🎲 Generator de Chestionare")
    st.markdown("Acest modul simuleaza corectarea unui set de lucrari. Extrage 5 intrebari random din baza de date.")

    if df_test is None:
        st.error("Nu gasesc fisierul data/test/test.csv!")
    else:
        num_questions = st.slider("Cate intrebari sa generam?", 1, 10, 5)

        if st.button("Genereaza si Corecteaza Chestionar", type="primary"):
            quiz_sample = df_test.sample(n=num_questions)
            results = []

            my_bar = st.progress(0)
            step = 0

            for idx, row in quiz_sample.iterrows():
                q_text = row.get('question_id', 'Intrebare')
                stud_ans = row['answer_student']
                corr_ans = row['answer_correct']
                real_grade = row['score_manual']

                pred_grade, pred_raw = predict_single(stud_ans, corr_ans)

                results.append({
                    "Intrebare": q_text,
                    "Raspuns Student": stud_ans,
                    "Raspuns Barem": corr_ans,
                    "Nota Reala": real_grade,
                    "Nota AI": pred_grade,
                    "Eroare": abs(real_grade - pred_grade)
                })

                step += 1
                my_bar.progress(int(step / num_questions * 100))

            my_bar.empty()

            st.success("✅ Corectare Finalizata!")

            res_df = pd.DataFrame(results)


            def color_rows(row):
                if row['Eroare'] == 0:
                    return ['background-color: #d4edda; color: black'] * len(row)  # Verde + Text Negru
                elif row['Eroare'] <= 1.0:
                    return ['background-color: #fff3cd; color: black'] * len(row)  # Galben + Text Negru
                else:
                    return ['background-color: #f8d7da; color: black'] * len(row)  # Rosu + Text Negru


            st.dataframe(res_df.style.apply(color_rows, axis=1), use_container_width=True)

            accuracy = len(res_df[res_df['Eroare'] == 0]) / num_questions * 100
            st.metric("Acuratete pe acest chestionar", f"{accuracy:.1f}%")