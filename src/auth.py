import streamlit as st
from src.config import ADMIN_PASSWORD


def check_password():
    """
    Shows a password input and returns True only if correct.
    Uses Streamlit session state so the user only needs to enter once.
    """
    if st.session_state.get("admin_authenticated"):
        return True

    st.markdown("""
    <div style="
        max-width: 400px;
        margin: 5rem auto;
        padding: 2.5rem;
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🔐</div>
        <h2 style="color: #1e3c72; margin-bottom: 0.5rem;">Admin Access</h2>
        <p style="color: #6b7280; font-size: 0.95rem;">This page is restricted to SKCET administrators.</p>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input("Enter admin password", type="password", key="admin_password_input")

    if st.button("🔓 Login", use_container_width=True):
        if password == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")

    return False
