import streamlit as st
from PyPDF2 import PdfReader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
OpenAI_API_KEY=""
st.header("NoteBot")
with st.sidebar:
 st.title("My Notes")
 file = st.file_uploader("Upload notes PDF and start asking questions", type="pdf")
# extracting the text from pdf file
if file is not None:
 my_pdf=PdfReader(file)
 text=""
 for page in my_pdf.pages:
    text += page.extract_text()
 #st.write(text)
 # break it into Chunks
 splitter = RecursiveCharacterTextSplitter(
chunk_size=300,chunk_overlap=50,length_function=len)
 chunks=splitter.split_text(text)
 #st.write(chunks)
 #creating Object of OpenAIEmbeddings class that let us connect with OpenAI's Embedding
 embeddings=OpenAIEmbeddings(api_key="sk-proj-vyjSlYCZCMhQ3Se4oSJgTfqQU0_A66DqIMkWC91DbP-93BbniuffC5baiSoqwYs2LYMA71oRrzT3BlbkFJrMxPavXl7fPbQu7CoLoOf63RjesTbkhf4XYlRbfy-7D"
                                    "JI1NE4AnhAlcospxuycXrGoUa6aBwIA")
 #Creating VectorDB & Storing embeddings into it
vector_store=FAISS.from_texts(chunks,embeddings)
 #get user query
user_query = st.text_input("Type your query here")
 #semantic search from vector store
if user_query:
 matching_chunks=vector_store.similarity_search(user_query)
 #define our LLM
 llm = ChatOpenAI(
 api_key=OpenAI_API_KEY,
 max_tokens=300,
 temperature=0,
 model="gpt-3.5-turbo"
 )
 #Approach 1 : Generate Response using load_qa_chain() function
 # chain=load_qa_chain(llm, chain_type="stuff")
 #output=chain.run(question=user_query, input_documents=matching_chunks)
 #st.write(output)
 # Approach 2 : Generate Response create_stuff_documents_chain() function
 customized_prompt = ChatPromptTemplate.from_template(
 """ You are my assistant tutor. Answer the question based on the following context and
 if you did not get the context simply say "I don't know Jenny" :
 {context}
 Question: {input}
 """
 )
 chain = create_stuff_documents_chain(llm, customized_prompt)
 output=chain.invoke({"input":user_query, "input_documents": matching_chunks})
 st.write(output)