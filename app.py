import helper
import streamlit as st

st.title("Tweet Analysis tool")
user_text = st.text_input(label = 'Write text', value="")
if st.button("pred"):
    pred,prob = helper.predict_spam(user_text)
    st.write(f"predition = {pred}, prob = {prob}")