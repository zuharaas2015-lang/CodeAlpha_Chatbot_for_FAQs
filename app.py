import streamlit as st

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
<style>

/* Main page */
.stApp {
    background-color: #07132d;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f1d3a;
    border-right: 1px solid #1e3a5f;
}
/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
/* Main Streamlit Containers */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a !important;
}

[data-testid="stMain"] {
    background-color: #0f172a !important;
}

.block-container {
    background-color: #0f172a !important;
}

/* Bottom Chat Area */
[data-testid="stBottomBlockContainer"] {
    background-color: #0f172a !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Title */
h1 {
    color: #60a5fa !important;
    text-align: center;
    margin-bottom: 0px !important;
}
/*TitleInfo*/
h2{
color: white;
text-align: center;}

/* Text */
p, label {
    color: white;
}

/* User Message */
[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #2563eb;
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 10px;
    color: white;
}

/* Bot Message */
[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #1e293b;
    padding: 12px;
    border-radius: 15px;
    margin-bottom: 10px;
    border: 1px solid #334155;
    color: white;
}

/* Chat Input */
.stChatInputContainer {
    background-color: #1e293b !important;
    border-radius: 10px;
}

[data-testid="stChatInput"] {
    background-color: #1e293b !important;
}

/* Sidebar Buttons */
.stButton button {
    width: 100%;
    border-radius: 10px;
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #334155 !important;
    font-weight: 500;
}
/* Remove Streamlit header */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Remove top padding */
.block-container {
    padding-top: 1rem !important;
}

/* Sidebar collapse arrow */
[data-testid="collapsedControl"] {
    color: white !important;
    opacity: 1 !important;
}

[data-testid="collapsedControl"] svg {
    width: 30px !important;
    height: 30px !important;
    stroke: white !important;
    stroke-width: 3px !important;
}

</style>
""", unsafe_allow_html=True)
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Load FAQ dataset
faq = pd.read_csv("faq.csv")

questions = faq['Question']
answers = faq['Answer']

# Text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    words = word_tokenize(text.lower())
    filtered = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

processed_questions = [preprocess(q) for q in questions]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

st.title("🤖 Smart FAQ Chatbot")

st.markdown("""
<div style="
    text-align:center;
    color:white;
    font-size:22px;
    font-weight:500;
    margin-top:-10px;
    margin-bottom:25px;
">
💬 Welcome! Ask me anything about our services.
</div>
""", unsafe_allow_html=True)
with st.sidebar:

    st.markdown("### 🤖 About")

    st.write(
        "This intelligent FAQ chatbot leverages Natural Language Processing (NLP) to understand user queries and deliver accurate, context-aware responses from a predefined knowledge base."
    )

    st.markdown("### Frequently Asked Questions")

    if st.button("What courses do you offer?"):
        st.session_state.clicked_question = "What courses do you offer?"

    if st.button("How can I contact support?"):
        st.session_state.clicked_question = "How can I contact support?"

    if st.button("Do you provide certificates?"):
        st.session_state.clicked_question = "Do you provide certificates?"

    if st.button("What are your working hours?"):
        st.session_state.clicked_question = "What are your working hours?"

    if st.button("How do I enroll in a course?"):
        st.session_state.clicked_question = "How do I enroll in a course?"

    if st.button("Is there a refund policy?"):
        st.session_state.clicked_question = "Is there a refund policy?"

    if st.button("Where are you located?"):
        st.session_state.clicked_question = "Where are you located?"

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_question = st.chat_input("Type your question here...")

if "clicked_question" in st.session_state:
    user_question = st.session_state.clicked_question
    del st.session_state.clicked_question

if "clicked_question" in st.session_state:
    user_question = st.session_state.clicked_question
    del st.session_state.clicked_question

if user_question:

    st.session_state.messages.append(("You", user_question))

    processed_input = preprocess(user_question)

    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence > 0.3:
        response = answers.iloc[best_match]
    else:
        response = "Sorry, I couldn't understand your question."

    st.session_state.messages.append(("Bot", response))

# Display messages
for sender, message in st.session_state.messages:

    if sender == "You":
      st.markdown(f"""
        <div style="
        display:flex;
        justify-content:flex-end;
        align-items:center;
        margin:12px 0;
        gap:8px;
       ">
        <div style="
            background:#7c3aed;
            color:white;
            padding:12px 18px;
            border-radius:18px;
            max-width:60%;
        ">
            {message}
        </div>
        <div style="font-size:24px;">👤</div>
     </div>
     """, unsafe_allow_html=True)
    else:
      st.markdown(f"""
        <div style="
        display:flex;
        justify-content:flex-start;
        align-items:center;
        margin:12px 0;
        gap:8px;
        ">
        <div style="
        font-size:30px;
        color:white;
     ">
      🤖
     </div>

        <div style="
            background:#1e293b;
            color:white;
            padding:12px 18px;
            border-radius:18px;
            max-width:60%;
            border:1px solid #334155;
        ">
            {message}
        </div>
       </div>
    """, unsafe_allow_html=True)