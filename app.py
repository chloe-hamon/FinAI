import streamlit as st
from config import ACTIONS

# ─────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FinAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CHARGEMENT AGENT (une seule fois)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def charger_agent():
    from agent import creer_agent
    return creer_agent()


# ─────────────────────────────────────────
# CSS CUSTOM — Grenat & Bleu Cyan
# ─────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0f0f1a;
        color: #e8e8e8;
    }
    [data-testid="stSidebar"] {
        background-color: #120a1f;
        border-right: 2px solid #6B2D6B;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00C9C8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6B2D6B;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .msg-user {
        background-color: #5a2070;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.95rem;
    }
    .msg-assistant {
        background-color: #0a1a3a;
        color: #e8e8e8;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        border-left: 3px solid #00C9C8;
        font-size: 0.95rem;
    }
    .stButton > button {
        background-color: #5a2070;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #00C9C8;
        color: #0f0f1a;
    }
    .stSelectbox label {
        color: #00C9C8 !important;
        font-weight: 600;
    }
    hr {
        border: 1px solid #5a2070;
        margin: 1rem 0;
    }
    .badge-action {
        background-color: #00C9C8;
        color: #0f0f1a;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TITRE PRINCIPAL
# ─────────────────────────────────────────
st.markdown("<div class='main-title'>🤖 FinAI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Votre analyste financier propulsé par l'intelligence artificielle</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────
# CHARGEMENT AVEC SPINNER VISIBLE
# ─────────────────────────────────────────
with st.spinner("⏳ Chargement du moteur FinAI..."):
    agent_executor, tools = charger_agent()

# ─────────────────────────────────────────
# INITIALISATION SESSION
# ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "suggestion_choisie" not in st.session_state:
    st.session_state.suggestion_choisie = None

if "action_selectionnee" not in st.session_state:
    st.session_state.action_selectionnee = list(ACTIONS.keys())[0]

if "agent_charge" not in st.session_state:
    st.session_state.agent_charge = False


# ─────────────────────────────────────────
# FONCTION APPEL AGENT
# ─────────────────────────────────────────
def interroger_agent(question: str) -> str:
    try:
        mots_analyse = ["score", "analyse", "opportunité", "recommandation", "vaut-il"]
        if any(mot in question.lower() for mot in mots_analyse):
            from agent import detecter_ticker
            ticker_detected = detecter_ticker(question)
            if ticker_detected:
                question = f"{question} — utilise analyser_action avec le ticker {ticker_detected}"

        resultat = agent_executor.invoke({"input": question})
        return resultat.get("output", "Pas de réponse.")
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 FinAI")
    st.markdown("*Votre analyste financier IA*")
    st.markdown("---")

    st.markdown("### 🎯 Actif analysé")
    action = st.selectbox(
        "Choisissez une action :",
        options=list(ACTIONS.keys()),
        format_func=lambda x: f"{x} — {ACTIONS[x]}",
        key="action_selectionnee"
    )

    st.markdown(
        f"<div class='badge-action'>📊 {action}</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🛠️ Outils actifs")
    if tools:
        for tool in tools:
            st.markdown(f"- `{tool.name}`")
    else:
        st.markdown("- *chargement...*")

    st.markdown("---")

    st.markdown("### 💡 Suggestions")
    suggestions = [
        f"Analyse {action}",
        f"Quelles sont les actualités récentes sur {action} ?",
        f"Quels sont les risques de {action} ?",
        f"Quel est le score global de {action} ?",
        f"Quel est le cours actuel de {action} ?",
        f"Performance de {action} sur 1 mois",
    ]

    for suggestion in suggestions:
        if st.button(suggestion, key=f"sug_{suggestion}"):
            st.session_state.suggestion_choisie = suggestion

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>FinAI v1.0 — Projet E3 2026<br>"
        f"💬 {len(st.session_state.messages)//2} échange(s) dans la session</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────
# AFFICHAGE HISTORIQUE CHAT
# ─────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='msg-user'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='msg-assistant'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────
# TRAITEMENT D'UNE QUESTION (fonction mutualisée)
# ─────────────────────────────────────────
def traiter_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f"<div class='msg-user'>🧑 {question}</div>",
        unsafe_allow_html=True
    )

    with st.spinner("🔍 Analyse en cours..."):
        reponse = interroger_agent(question)

    st.session_state.messages.append({"role": "assistant", "content": reponse})
    st.markdown(
        f"<div class='msg-assistant'>🤖 {reponse}</div>",
        unsafe_allow_html=True
    )
    st.rerun()


# ─────────────────────────────────────────
# GESTION SUGGESTION CLIQUÉE
# ─────────────────────────────────────────
if st.session_state.suggestion_choisie:
    question = st.session_state.suggestion_choisie
    st.session_state.suggestion_choisie = None
    traiter_question(question)


# ─────────────────────────────────────────
# INPUT UTILISATEUR
# ─────────────────────────────────────────
question = st.chat_input("Posez votre question financière...")

if question:
    traiter_question(question)
