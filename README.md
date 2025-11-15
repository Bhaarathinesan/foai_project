Spam Detector using Bayes Theorem

A modern Streamlit web application that uses Bayes Theorem + Naive Bayes to estimate whether an email is spam.
This project demonstrates how Bayesian probability can be applied in real-world text classification.

🚀 Live Demo (Optional)

If hosted on Streamlit Cloud or HuggingFace, put your link here:

https://your-app-link.streamlit.app/

🧠 Project Overview

This project visualizes how Bayes Theorem can be used to compute the probability that an email is spam based on the words it contains.

✔ How it works:

A small dataset of spam & ham messages is preloaded.

Words are tokenized and counted.

Using Naive Bayes (bag-of-words):

Likelihood = frequency of words in spam vs ham.

Priors assumed equal (50–50).

Bayes theorem is applied:

𝑃
(
𝑆
𝑝
𝑎
𝑚
∣
𝑊
𝑜
𝑟
𝑑
𝑠
)
=
𝑃
(
𝑊
𝑜
𝑟
𝑑
𝑠
∣
𝑆
𝑝
𝑎
𝑚
)
⋅
𝑃
(
𝑆
𝑝
𝑎
𝑚
)
𝑃
(
𝑊
𝑜
𝑟
𝑑
𝑠
)
P(Spam∣Words)=
P(Words)
P(Words∣Spam)⋅P(Spam)
	​

✔ What the app provides:

Modern UI with cards and probability indicators

Classification based on text input

Bar graph visualization of spam probability

Simple, clear explanation of Bayesian reasoning
