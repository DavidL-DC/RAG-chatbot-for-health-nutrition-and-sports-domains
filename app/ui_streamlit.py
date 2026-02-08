import streamlit as st
from query import ask, ABSTENTION_TEXT

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 RAG-Chatbot (Ernährung • Gesundheit • Sport)")
st.caption(
    "Antwortet ausschließlich auf Basis deines wissenschaftlichen Korpus (mit Quellen)."
)

question = st.text_input(
    "Frage eingeben",
    placeholder="z. B. Welche gesundheitlichen Effekte hat regelmäßige körperliche Aktivität?",
)

run = st.button("Antwort generieren", type="primary")

if run:
    if not question.strip():
        st.warning("Bitte gib eine Frage ein.")
    else:
        with st.spinner("Suche relevante Quellen und generiere Antwort..."):
            result = ask(question.strip())

        st.subheader("Antwort")
        st.write(result["answer"])

        if result["abstain"] or result["answer"].strip() == ABSTENTION_TEXT:
            st.info(
                "Hinweis: Keine ausreichend relevante Kontextbasis gefunden (Abstention)."
            )
        else:
            st.subheader("Quellen")
            st.code(result["sources_str"], language="text")

st.divider()
st.caption(
    "Tipp: Wenn du Ergebnisse vergleichen willst, stelle dieselbe Frage mehrfach und beobachte Quellen/Abstention."
)
