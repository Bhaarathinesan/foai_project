import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------- STREAMLIT PAGE CONFIG -----------------------
st.set_page_config(
    page_title="Spam Classifier using Bayes Theorem",
    layout="wide",
    page_icon="📧",
)

# ----------------------- CUSTOM UI STYLES -----------------------
st.markdown("""
<style>
    .title { font-size:32px !important; font-weight:700; color:#1E88E5; }
    .card {
        padding:20px; border-radius:15px; background:#ffffff;
        box-shadow:0px 4px 10px rgba(0,0,0,0.07); border:1px solid #eee;
        margin-top:10px;
    }
    .result-card {
        padding:25px; border-radius:20px; background:#F7F9FC;
        box-shadow:0px 6px 16px rgba(0,0,0,0.08);
        text-align:center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📧 Spam Detector using Bayes Theorem</div>",
            unsafe_allow_html=True)

st.write("This app uses **Bayes Theorem + Naive Bayes** to estimate whether an email is spam.")


# ----------------------- SAMPLE TRAINING DATA -----------------------

spam_samples = [
    "win money now", "free cash offer", "claim your lottery prize",
    "winner prize cash", "free vacation now"
]

ham_samples = [
    "meeting schedule attached", "project deadline extended",
    "please review the report", "let’s have a call tomorrow",
    "team lunch invitation"
]

def tokenize(text):
    return text.lower().split()

spam_words = []
ham_words = []

for s in spam_samples:
    spam_words.extend(tokenize(s))
for h in ham_samples:
    ham_words.extend(tokenize(h))

# Word counts
spam_counts = Counter(spam_words)
ham_counts = Counter(ham_words)

# Total words
total_spam_words = sum(spam_counts.values())
total_ham_words = sum(ham_counts.values())

# Priors
P_spam = 0.5
P_ham = 0.5

# ----------------------- USER INPUT -----------------------

email_text = st.text_area("📩 Enter Email Content", height=150)

if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter an email to analyze.")
    else:
        words = tokenize(email_text)

        # Likelihood calculation (Naive Bayes)
        P_words_given_spam = 1
        P_words_given_ham = 1

        for word in words:
            # Laplace smoothing
            P_words_given_spam *= (spam_counts[word] + 1) / (total_spam_words + 100)
            P_words_given_ham *= (ham_counts[word] + 1) / (total_ham_words + 100)

        # Bayes theorem
        num = P_words_given_spam * P_spam
        den = (P_words_given_spam * P_spam) + (P_words_given_ham * P_ham)
        posterior_spam = num / den

        # ----------------------- RESULT UI -----------------------
        st.markdown("<h3>📊 Result</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
                <div class='card'>
                    <h4>Probability of Spam</h4>
                    <h2 style="color:#D32F2F">{posterior_spam*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class='card'>
                    <h4>Probability of Not Spam</h4>
                    <h2 style="color:#388E3C">{(1-posterior_spam)*100:.2f}%</h2>
                </div>
            """, unsafe_allow_html=True)

        # ----------------------- CHART -----------------------
        st.markdown("### 📈 Probability Visualization")

        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(["Spam", "Not Spam"], [posterior_spam, 1 - posterior_spam])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # ----------------------- EXPLANATION -----------------------
        st.write("---")
        st.info("""
### 🧠 How This Works
Using Bayes Theorem:

\[
P(Spam | Words) = \frac{P(Words | Spam) \cdot P(Spam)}{P(Words)}
\]

The model:
- Counts how frequently each word appears in spam vs normal emails  
- Computes likelihoods using Naive Bayes  
- Uses Bayes theorem to get the final spam probability  
""")


