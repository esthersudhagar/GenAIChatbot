import streamlit as st
from streamlit import sidebar
st.header("NoteBot")
with sidebar:
    st.title("Esther Notebot")
    file=st.file_uploader("Upload notes PDF and start asking questions",type=["pdf"])