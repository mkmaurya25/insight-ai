import os
import streamlit as st
import pickle
import time
from langchain_groq import ChatGroq
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Streamlit UI
st.title("InsightAI: News Research🔎")
st.sidebar.title("API Configuration")

# Input for API key (hidden with password field)
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# If API key is not provided, show a warning message
if not api_key:
    st.sidebar.warning("Please enter your GROQ API Key to proceed.")
else:
    st.sidebar.success("API Key entered successfully!")

# Input for URLs
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# Ensure API key is passed to the ChatGroq client
if api_key:
    # Create the ChatGroq client with the API key
    llm = ChatGroq(model="llama3-8b-8192", api_key=api_key, max_tokens=500)

    if process_url_clicked:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # Create embeddings and FAISS index
        embeddings = HuggingFaceEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

    # Handle the query
    query = main_placeholder.text_input("Question: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=False)

                # Display the answer and sources
                st.header("Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)

else:
    # Block further processing if the API key is missing
    st.warning("Please provide the API key to process URLs and query the model.")