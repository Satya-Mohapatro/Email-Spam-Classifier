import streamlit as st
import joblib

pipeline = joblib.load('svm_spam_model.pkl')

st.title("📨 Email/SMS Spam Classifier using SVM")
st.write("Paste your message below to check if it is **Spam** or **Not Spam**.")

user_input = st.text_area("Enter email or SMS text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text for prediction.")
    else:
        prediction = pipeline.predict([user_input])[0]
        prediction_proba = pipeline.predict_proba([user_input])[0][prediction]

        if prediction == 1:
            st.error(f"🚫 This message is predicted as **SPAM** with probability {prediction_proba:.2f}.")
        else:
            st.success(f"✅ This message is predicted as **NOT SPAM** with probability {prediction_proba:.2f}.")
