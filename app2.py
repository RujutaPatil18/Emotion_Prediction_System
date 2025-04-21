# app2.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping
id2label = model.config.id2label

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "joy": "😂", "sadness": "😔",
    "surprise": "😮", "neutral": "😐"
}

# Predict emotion
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_pred = torch.argmax(probs, dim=1).item()
    emotion = id2label[top_pred]
    confidence = probs[0, top_pred].item()
    return emotion, confidence, probs[0].numpy()

# Streamlit app
def main():
    st.title("Emotion Prediction System 😄😔")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home - Emotion In Text")

        with st.form(key='emotion_form'):
            raw_text = st.text_area("Type something here...")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and raw_text:
            emotion, confidence, prob_array = predict_emotions(raw_text)
            col1, col2 = st.columns(2)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(emotion, "")
                st.write(f"{emotion}: {emoji_icon}")
                st.write(f"Confidence: {confidence:.2f}")

            with col2:
                st.success("Prediction Probability")
                labels = [id2label[i] for i in range(len(prob_array))]
                proba_df = pd.DataFrame({
                    "emotions": labels,
                    "probability": prob_array
                })

                fig = alt.Chart(proba_df).mark_bar().encode(
                    x='emotions', y='probability', color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)
    elif choice == "Monitor":
        st.subheader("Monitor App")
        st.markdown("### Sample Prediction Logs")

        sample_data = {
            "Text": ["I am so happy today!", "I'm scared of this exam", "He made me angry"],
            "Predicted Emotion": ["joy", "fear", "anger"],
            "Confidence": [0.88, 0.75, 0.81]
        }

        st.table(pd.DataFrame(sample_data))

        st.markdown("---")
        st.markdown("### Model Performance")
        st.write("**Model Used:** DistilBERT fine-tuned for Emotion Classification")
        st.write("**Accuracy (Previous ML Model):** 77% (Trained using XGBoost in Jupyter Notebook)")
        st.write("**Current Model:** Pretrained Transformer (`distilbert-base-uncased-emotion`)")
        st.write("**Advantage:** Improved prediction quality, generalization, and emoji support ✨")

    else:
        st.subheader("About")
        st.markdown("""
        ### Emotion Prediction System 😄😔😠  
        This mini project analyzes user input text and predicts the underlying emotion using natural language processing and deep learning.  
        The system provides a user-friendly interface and visualizes prediction probabilities, helping demonstrate the power of AI in understanding human emotion.

        ---
        ### 🔧 Technologies Used:
        - **Model:** `distilbert-base-uncased-emotion` (pretrained transformer)
        - **Frontend:** Streamlit
        - **Libraries:** `transformers`, `torch`, `pandas`, `Altair`, `numpy`

        ---
        ### 📊 Features:
        - Real-time emotion detection from user input
        - Emojis for better emotional expression
        - Confidence score + bar graph visualization
        - Clean UI and multi-tab layout

        ---
        ### 👥 Team Members:
        - **Rujuta Patil** (Roll No: 71)
        - **Gunjan Karekar** (Roll No: 79)
        - **Akanksha Chodankar** (Roll No: 65)
        - **Gayatri Machale ** (Roll No: 72)

        ---
        ### 🧠 Why Transformers?
        Transformers like BERT are state-of-the-art in NLP. We switched from traditional ML models to a pretrained transformer to boost accuracy, eliminate the need for manual training, and simplify deployment.

        ---
        ### 📍 Project Goal:
        To demonstrate the application of AI in real-world emotion analysis with an interactive and educational UI.
        """)



if __name__ == '__main__':
    main()
