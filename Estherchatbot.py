import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Load the key from an environment variable — set GOOGLE_API_KEY in Render's
# Environment tab, never hardcode it here.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

st.header("NoteBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")

# extracting the text from pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()

    # break it into Chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)

    # creating Object of GoogleGenerativeAIEmbeddings class that lets us connect with Gemini's Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    # Creating VectorDB & Storing embeddings into it
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user query
    user_query = st.text_input("Type your query here")

    # semantic search from vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        # define our LLM
        llm = ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gemini-flash-latest",
        )

        customized_prompt = ChatPromptTemplate.from_template(
            """ You are my assistant tutor. Answer the question based on the following context and
            if you did not get the context simply say "I don't know Jenny" :
            {context}
            Question: {input}
            """
        )
        chain = create_stuff_documents_chain(llm, customized_prompt)
        output = chain.invoke({"input": user_query, "context": matching_chunks})
        st.write(output)
