# InsightAI: An App for News Research with Language Models
In this article, we'll walk through the creation of a Streamlit application that leverages large language models for real-time research on news articles. Using the powerful LangChain framework, we'll create a system that allows users to input news URLs, generate embeddings from the text, and ask questions based on the retrieved information. The app integrates ChatGroq (a LLaMA-based language model) and Hugging Face embeddings, providing a scalable and responsive solution for working with textual data.

## Key Features of the App

- API Key Authentication: The app requires users to securely enter their GROQ API key.
- News Article Processing: Users can input URLs of news articles, and the app fetches, processes, and   splits the data into chunks for efficient information retrieval.
- Embeddings and Vector Search: The text from the articles is embedded using Hugging Face models, and stored in a FAISS vectorstore for fast similarity-based search.
- Question-Answering System: Users can ask questions about the content of the articles, and the system will retrieve and synthesize answers from the embedded vectors, powered by ChatGroq.

Let's dive into the details of how this works.

## Step 1: Setting up the Environment

The core libraries and tools used in this app are:

-   Streamlit: A Python framework for creating interactive, real-time web applications.
-   LangChain: A framework that enables building chains for connecting LLMs (Large Language Models) to a variety of data sources.
-   GROQ API: Provides access to the LLaMA-based ChatGroq language model.
-   Hugging Face Embeddings: Used to convert the textual data into vector representations.
-   FAISS: A library for efficient similarity search and clustering of dense vectors, used for storing and querying the document embeddings.

Before starting, install the required dependencies:
```
pip install streamlit langchain langchain_groq langchain_huggingface langchain_community unstructured faiss-cpu
```
You should have groq api key. You can get it from `https://console.groq.com`. You can also get the llm models from `https://console.groq.com/docs/models`.

## Step 2: Application Structure
### API Configuration

The app starts by asking the user to provide their GROQ API key. This is important because the language model requires an API key for authentication. For security purposes, we use a password-protected input field to hide the API key:

```
# Input for API key (hidden with password field)
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

# If API key is not provided, show a warning message
if not api_key:
    st.sidebar.warning("Please enter your GROQ API Key to proceed.")
else:
    st.sidebar.success("API Key entered successfully!")
```
This block ensures the user cannot proceed with the URL processing or question answering functionality unless they provide the API key. Alternativley, you can create a .env file in root directory of the project and use
```
from dotenv import load_dotenv
load_dotenv()
```
The .env file will look like
```
GROQ_API_KEY = "your api key here"
```
## URL Input and Data Processing

Next, users can input up to three news article URLs. The app uses UnstructuredURLLoader from LangChain to fetch and load the content of the URLs.
```
# Input for URLs
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
```
Once the URLs are entered, the process_url_clicked button initiates the data retrieval and preprocessing steps. The content from the URLs is split into manageable chunks using RecursiveCharacterTextSplitter:
```
if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
```
This chunking step is essential for improving the accuracy of the language model's question-answering ability and optimizing the performance of the vector search.

## Creating Embeddings and Storing with FAISS

With the document chunks ready, the next step is to create embeddings. We use the Hugging Face Embeddings to transform the text into vectors, which can then be efficiently searched using FAISS.
```
# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings()
vectorstore_openai = FAISS.from_documents(docs, embeddings)
```
The resulting FAISS index is stored in a pickle file for reuse:
```
with open(file_path, "wb") as f:
    pickle.dump(vectorstore_openai, f)
```
This approach ensures that once the embeddings are created, they don't need to be recalculated each time the app is run, speeding up future queries.

### Question-Answering with RetrievalQA

The final core feature of the app is the ability for users to ask questions based on the content of the news articles. We use RetrievalQAWithSourcesChain to combine the language model with the FAISS retriever:
```
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=False)
```
This provides users with context on where the information is coming from, allowing them to verify the credibility of the sources used by the model.

## Step 3: Deploying the App

To deploy the app, you can easily use Streamlit Cloud. Here’s a quick guide on how to get the app running:

-   Create a GitHub Repository: Push the app's code to a GitHub repository.

-   Streamlit Cloud Deployment: Go to Streamlit Cloud and connect your GitHub repository. Streamlit Cloud will automatically detect the streamlit_app.py file and deploy the app for you.

-   Environment Variables: If you'd rather not require the user to enter the API key every time, you can set the GROQ_API_KEY as an environment variable in the Streamlit Cloud settings.

-   Accessing the App: Once deployed, users can access the app through the provided Streamlit Cloud URL.

## Conclusion

This article demonstrated how to build a Streamlit app that allows users to research news articles using language models. By integrating LangChain with ChatGroq and FAISS, we’ve created a system that allows users to input URLs, split content into chunks, embed the text for fast retrieval, and query the model for answers.

This approach can be extended to other domains, making it a powerful framework for real-time data analysis and question-answering from unstructured data. I have used streamlit to build the interface, you can use gradio or any other tool of your choice.

Feel free to fork the project, customize it with additional features, and deploy it to fit your specific needs. Happy coding!