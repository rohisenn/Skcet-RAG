import warnings
warnings.filterwarnings('ignore')
import time
import json as _json
import html as _html

import streamlit as st
from src.embeddings import get_embedding_model
from src.vector_store import load_vector_store
from src.retriever import get_retriever
from src.rag import run_rag_stream, finalize_rag
from src.memory import ConversationMemory
from src.loader import load_documents
from src.database import init_db, log_query, update_rating, flag_query

init_db()

# ── Rate Limit Config ────────────────────────────────────────────────────────
RATE_LIMIT = 20          # max queries per session

def action_buttons(text: str) -> str:
    """Returns safe HTML for copy-to-clipboard and TTS listen buttons."""
    safe_json = _json.dumps(text)
    safe_html = _html.escape(safe_json)
    return f"""
    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.4rem;">
        <button
            onclick="navigator.clipboard.writeText({safe_html}).then(()=>{{this.innerText='\u2705 Copied!';setTimeout(()=>this.innerText='\U0001f4cb Copy',1500)}}).catch(()=>{{}})"
            style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);color:#a5b4fc;
                   border-radius:8px;padding:0.25rem 0.65rem;font-size:0.75rem;cursor:pointer;
                   font-family:Inter,sans-serif;transition:all 0.2s"
            onmouseover="this.style.background='rgba(99,102,241,0.25)'"
            onmouseout="this.style.background='rgba(99,102,241,0.1)'">
            \U0001f4cb Copy
        </button>
        <button
            onclick="let u = new SpeechSynthesisUtterance({safe_html}); window.speechSynthesis.cancel(); window.speechSynthesis.speak(u); this.innerText='🔊 Playing...'; u.onend=()=>this.innerText='🔊 Listen';"
            style="background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.3);color:#e879f9;
                   border-radius:8px;padding:0.25rem 0.65rem;font-size:0.75rem;cursor:pointer;
                   font-family:Inter,sans-serif;transition:all 0.2s"
            onmouseover="this.style.background='rgba(168,85,247,0.25)'"
            onmouseout="this.style.background='rgba(168,85,247,0.1)'">
            🔊 Listen
        </button>
    </div>"""

st.set_page_config(
    page_title="SKCET AI Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ─── Reset & Base ─────────────────────────────── */
    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

    html, body, [class*="css"] {
        color-scheme: dark;
    }

    /* ─── App Background ────────────────────────────── */
    .stApp {
        background: #0a0a0f;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99,102,241,0.25) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 80% 80%, rgba(168,85,247,0.15) 0%, transparent 60%);
    }

    /* ─── Hide Streamlit chrome ─────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 5rem !important; }

    /* ─── Sidebar ───────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.98) !important;
        border-right: 1px solid rgba(99,102,241,0.2) !important;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stDownloadButton button {
        background: rgba(99,102,241,0.15) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        color: #a5b4fc !important;
        border-radius: 10px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s !important;
    }
    [data-testid="stSidebar"] .stDownloadButton button:hover {
        background: rgba(99,102,241,0.3) !important;
        border-color: #6366f1 !important;
    }

    /* ─── Header Card ───────────────────────────────── */
    .hero-card {
        background: linear-gradient(135deg,
            rgba(99,102,241,0.12) 0%,
            rgba(168,85,247,0.12) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(12px);
    }
    .hero-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(99,102,241,0.05), transparent);
        pointer-events: none;
    }
    .hero-icon {
        font-size: 2.8rem;
        margin-bottom: 0.4rem;
        display: block;
        filter: drop-shadow(0 0 20px rgba(99,102,241,0.6));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.25rem;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 400;
        margin: 0 0 1rem;
    }
    .hero-pills {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .pill {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* ─── Chat Messages ─────────────────────────────── */
    .stChatMessage { background: transparent !important; }

    /* User bubble */
    .stChatMessage:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #f1f5f9 !important;
        border-radius: 18px 18px 4px 18px !important;
        box-shadow: 0 4px 20px rgba(79,70,229,0.35) !important;
        padding: 0.85rem 1.2rem !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        max-width: 82% !important;
        margin-left: auto !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    /* Assistant bubble */
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: rgba(15, 18, 35, 0.85) !important;
        color: #e2e8f0 !important;
        border-radius: 18px 18px 18px 4px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
        padding: 0.85rem 1.2rem !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Avatars */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border-radius: 50% !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.5) !important;
    }
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #0f172a, #1e1b4b) !important;
        border-radius: 50% !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.3) !important;
    }

    /* ─── Expander (Sources) ────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(15, 18, 35, 0.6) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        border-radius: 12px !important;
        margin-top: 0.5rem !important;
    }
    [data-testid="stExpander"] summary { color: #a5b4fc !important; font-size: 0.85rem !important; }
    [data-testid="stExpander"] p { color: #94a3b8 !important; font-size: 0.85rem !important; }

    /* ─── Chat Input ────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        background: rgba(15, 18, 35, 0.9) !important;
        border: 1.5px solid rgba(99,102,241,0.35) !important;
        border-radius: 16px !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.05),
                    0 4px 20px rgba(0,0,0,0.3) !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.2),
                    0 4px 20px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }

    /* ─── Follow-up buttons ─────────────────────────── */
    .stButton button {
        background: rgba(99,102,241,0.1) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: #a5b4fc !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.4rem 0.75rem !important;
        transition: all 0.2s ease !important;
        white-space: normal !important;
        text-align: left !important;
        line-height: 1.4 !important;
    }
    .stButton button:hover {
        background: rgba(99,102,241,0.25) !important;
        border-color: #6366f1 !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ─── Spinner ───────────────────────────────────── */
    [data-testid="stSpinner"] {
        color: #6366f1 !important;
    }

    /* ─── Caption (confidence badges) ──────────────── */
    .stChatMessage small, .stChatMessage .stCaption {
        color: #64748b !important;
        font-size: 0.78rem !important;
    }

    /* ─── Scrollbar ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.4);
        border-radius: 99px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.65); }

</style>
""", unsafe_allow_html=True)

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <span class="hero-icon">🎓</span>
    <h1 class="hero-title">SKCET AI Assistant</h1>
    <p class="hero-subtitle">Sri Krishna College of Engineering and Technology</p>
    <div class="hero-pills">
        <span class="pill">⚡ Powered by Groq</span>
        <span class="pill">🧠 RAG Architecture</span>
        <span class="pill">📚 Smart Citations</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ── Load RAG ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    docs = load_documents()
    embedding_model = get_embedding_model()
    vectordb = load_vector_store(embedding_model)
    retriever = get_retriever(vectordb, docs)
    return retriever

with st.spinner("⚡ Initialising knowledge base..."):
    retriever = load_rag()

# ── Sidebar: Tone + Quick Questions + Export ──────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Language")
    language = st.selectbox(
        label="language",
        options=["English", "Tamil", "Hindi", "Malayalam", "Telugu"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state["language"] = language

    st.markdown("---")
    st.markdown("### 🎨 Response Style")
    tone = st.radio(
        label="tone",
        options=["Detailed", "Concise", "Bullet Points"],
        index=0,
        horizontal=False,
        label_visibility="collapsed"
    )
    st.session_state["tone"] = tone

    st.markdown("---")
    st.markdown("### ⚡ Quick Questions")
    quick_questions = [
        "What are the courses offered at SKCET?",
        "How is the placement record at SKCET?",
        "Who is the Principal of SKCET?",
        "What are the hostel facilities?",
        "What is the fee structure?",
        "Tell me about sports at SKCET",
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state["pending_followup"] = q
            st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Export Chat")
    if st.session_state.get("messages"):
        lines = []
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
            lines.append("")
        chat_text = "\n".join(lines)
        st.download_button(
            label="Download Conversation (.txt)",
            data=chat_text,
            file_name="skcet_chat_export.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.caption("No conversation to export yet.")

    st.markdown("---")
    remaining = max(0, RATE_LIMIT - st.session_state.query_count)
    st.caption(f"📊 {st.session_state.query_count}/{RATE_LIMIT} queries used this session")
    st.progress(st.session_state.query_count / RATE_LIMIT)


# ── Helpers ──────────────────────────────────────────────────────────────────
def handle_feedback(query_id, feedback_value):
    rating = "thumbs_up" if feedback_value == 1 else "thumbs_down"
    update_rating(query_id, rating)

CONFIDENCE_BADGES = {
    "High":   "🟢 High Confidence",
    "Medium": "🟡 Medium Confidence",
    "Low":    "🔴 Low Confidence",
}

# ── Chat History ─────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if "confidence" in msg and "response_time_ms" in msg:
                badge = CONFIDENCE_BADGES.get(msg["confidence"], "")
                tstr = msg.get("tone", "")
                tone_icon = {"Detailed": "📝", "Concise": "⚡", "Bullet Points": "📌"}.get(tstr, "")
                st.caption(f"{badge} · ⏱️ {msg['response_time_ms']}ms {tone_icon}")
            elif "confidence" in msg:
                st.caption(CONFIDENCE_BADGES.get(msg["confidence"], ""))
            st.markdown(action_buttons(msg["content"]), unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📚 Source Documents"):
                    for source in msg["sources"]:
                        st.markdown(f"- `{source}`")
            if "query_id" in msg:
                feedback_key = f"feedback_{msg['query_id']}"
                st.feedback(
                    "thumbs",
                    key=feedback_key,
                    on_change=handle_feedback,
                    args=(msg["query_id"], st.session_state.get(feedback_key))
                )
                if st.button("🚩 Report Error", key=f"flag_{msg['query_id']}", help="Flag this answer as incorrect"):
                    flag_query(msg["query_id"])
                    st.toast("Answer flagged for admin review! Thank you.", icon="🚩")
            followups = msg.get("followups", [])
            if followups:
                st.markdown(
                    "<p style='color:#64748b;font-size:0.8rem;margin:0.6rem 0 0.4rem;'>💡 Suggested follow-ups</p>",
                    unsafe_allow_html=True
                )
                cols = st.columns(len(followups))
                for i, fq in enumerate(followups):
                    with cols[i]:
                        if st.button(fq, key=f"fq_{idx}_{i}", use_container_width=True):
                            st.session_state["pending_followup"] = fq
                            st.rerun()

# ── Welcome message ───────────────────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm your **SKCET Knowledge Assistant**. Ask me anything about courses, "
            "facilities, admissions, placements, faculty, and more! 🎓"
        )

# ── Chat Input ────────────────────────────────────────────────────────────────
if "pending_followup" in st.session_state:
    pending = st.session_state.pop("pending_followup")
else:
    pending = None

user_input = st.chat_input("Ask anything about SKCET...") or pending

if user_input:
    # ── Rate Limit Check ─────────────────────────────────────────────────────
    if st.session_state.query_count >= RATE_LIMIT:
        st.warning(
            f"🚫 You've reached the session limit of **{RATE_LIMIT} queries**. "
            "Please refresh the page to start a new session."
        )
        st.stop()

    selected_tone = st.session_state.get("tone", "Detailed")
    selected_lang = st.session_state.get("language", "English")
    st.session_state.query_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        t_start = time.time()
        stream_gen, sources, confidence = run_rag_stream(
            user_input, retriever, st.session_state.memory, tone=selected_tone, language=selected_lang
        )
        # ✨ Live streaming – the most noticeable UX feature!
        answer = st.write_stream(stream_gen)
        response_time_ms = int((time.time() - t_start) * 1000)

        badge = CONFIDENCE_BADGES.get(confidence, "")
        tone_icon = {"📝Detailed": "Detailed", "Concise": "⚡", "Bullet Points": "📌"}.get(selected_tone, "")
        st.caption(f"{badge} · ⏱️ {response_time_ms}ms")

        st.markdown(action_buttons(answer), unsafe_allow_html=True)

        if sources:
            with st.expander("📚 Source Documents"):
                for source in sources:
                    st.markdown(f"- `{source}`")

    # After streaming completes, save to memory and get follow-ups
    followups = finalize_rag(user_input, answer, st.session_state.memory)
    query_id = log_query(user_input, answer, response_time_ms, confidence)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence,
        "followups": followups,
        "query_id": query_id,
        "response_time_ms": response_time_ms,
        "tone": selected_tone,
    })
    st.rerun()