import streamlit as st
import pandas as pd
from collections import Counter
import re

from src.auth import check_password
from src.database import get_all_queries

st.set_page_config(
    page_title="FAQ Generator - SKCET Admin",
    page_icon="❓",
    layout="wide",
)

if not check_password():
    st.stop()

st.title("❓ FAQ Generator")
st.markdown("Auto-generate a FAQ document from the most frequently asked student queries.")

with st.spinner("Analyzing query database..."):
    queries = get_all_queries()

if not queries:
    st.info("No queries found in the database yet. Students need to ask some questions first!")
    st.stop()


def extract_keywords(text):
    """Extract 2-3 word n-grams from a query for clustering."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stopwords = {'what', 'is', 'the', 'are', 'how', 'do', 'does', 'a', 'an',
                 'in', 'of', 'to', 'for', 'at', 'about', 'and', 'i', 'can',
                 'me', 'tell', 'please', 'skcet', 'college'}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords


# Build keyword frequency counter
keyword_counter = Counter()
query_map = {}  # keyword -> first representative query+answer pair

for row in queries:
    user_input = row.get("user_input", "")
    keywords = extract_keywords(user_input)
    for kw in keywords:
        keyword_counter[kw] += 1
        if kw not in query_map:
            query_map[kw] = row

top_keywords = keyword_counter.most_common(15)

# ── Display Top FAQs Table ─────────────────────────────────────────────────────
st.subheader("📊 Top Query Topics")
st.markdown("Based on the most repeated keywords across all student queries.")

table_data = []
for kw, count in top_keywords:
    rep = query_map.get(kw, {})
    table_data.append({
        "Topic Keyword": kw.title(),
        "Times Asked": count,
        "Example Question": rep.get("user_input", "N/A")[:80] + "..." if len(rep.get("user_input", "")) > 80 else rep.get("user_input", "N/A")
    })

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Generate FAQ Markdown ─────────────────────────────────────────────────────
st.subheader("📄 Generated FAQ Document")

faq_lines = ["# SKCET Knowledge Assistant - Frequently Asked Questions\n",
             "_Auto-generated from student query patterns._\n",
             "---\n"]

for i, (kw, _) in enumerate(top_keywords[:10], 1):
    rep = query_map.get(kw, {})
    question = rep.get("user_input", f"Questions about {kw.title()}")
    answer = rep.get("assistant_response", "Please refer to the SKCET Knowledge Assistant for detailed information.")
    faq_lines.append(f"## Q{i}. {question}\n")
    faq_lines.append(f"{answer}\n")
    faq_lines.append("---\n")

faq_md = "\n".join(faq_lines)

st.markdown(faq_md[:2000] + "\n\n_...preview truncated. Download to see the full FAQ._")

st.download_button(
    label="📥 Download Full FAQ (.md)",
    data=faq_md,
    file_name="skcet_faq.md",
    mime="text/markdown",
    type="primary",
    use_container_width=True
)
