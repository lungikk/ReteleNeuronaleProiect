import streamlit as st
import joblib
import os
import numpy as np

st.set_page_config(page_title="Sistem Notare Automata")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

@st.cache_resource
def load_brain():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vect = joblib.load(VECTORIZER_PATH)
    return model, vect


model, vectorizer = load_brain()

st.title("Sistem AI de Notare Automata")
st.markdown(
    "Acest sistem foloseste o **Retea Neuronala (MLP)** antrenata pe text pentru a nota raspunsurile studentilor.")

if model is None:
    st.error("Eroare: Nu gasesc modelul 'trained_model.pkl'. Ruleaza intai train_final.py!")
else:
    st.success("Model neuronal incarcat cu succes!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Barem (Raspuns Corect)")
        correct_answer = st.text_area("Ce trebuia sa scrie studentul?",
                                      "Reteaua neuronala este un model inspirat din creierul uman.", height=150)

    with col2:
        st.subheader("Raspuns Student")
        student_answer = st.text_area("Ce a scris studentul?", "Este un model matematic inspirat biologic.", height=150)

    if st.button("Calculeaza Nota", type="primary"):
        if not student_answer or not correct_answer:
            st.warning("Te rog completeaza ambele campuri!")
        else:
            text_input = str(student_answer) + " " + str(correct_answer)

            vector_input = vectorizer.transform([text_input])

            vector_dense = vector_input.toarray()
            vector_final = np.array(vector_dense, dtype=np.float32)

            prediction = model.predict(vector_final)[0]

            possible_grades = np.array([0.0, 2.5, 4.0, 5.0])
            idx = (np.abs(possible_grades - prediction)).argmin()
            final_grade = possible_grades[idx]

            st.divider()
            col_res1, col_res2 = st.columns([1, 2])

            with col_res1:
                st.metric(label="Nota Calculata", value=str(final_grade))

            with col_res2:
                if final_grade == 5.0:
                    st.success("Excelent! Raspuns complet.")
                elif final_grade >= 4.0:
                    st.info("Raspuns Bun (Parafrazat).")
                elif final_grade >= 2.5:
                    st.warning("Raspuns Partial.")
                else:
                    st.error("Raspuns Gresit sau Irelevant.")

            st.caption(f"Scor brut (Raw Output): {prediction:.4f}")
