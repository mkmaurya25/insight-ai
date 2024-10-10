# InsightAI: News Research Tool 🔎

## Overview

**InsightAI** is a powerful news research tool that allows users to extract and analyze information from news articles using advanced language models. By providing URLs of news articles, users can ask questions about the content and receive detailed answers along with the sources of information.

## Features

- **Load news articles** from multiple URLs.
- **Process and split** article content into manageable chunks.
- **Embed text** using Hugging Face models for semantic search.
- **Utilize ChatGroq** for generating answers to user queries.
- **Display sources** of information alongside answers.

## Requirements

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Streamlit
- Langchain
- Hugging Face Transformers
- Groq Language Model
- dotenv

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/insightai.git
   cd insightai
   ```
2. **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

   Create a .env file in the root directory of the project and add your environment variables, such as API keys, if necessary.
    
   ```
   # Example .env content
   OPENAI_API_KEY="your_api_key_here"
   ```
4. **Run the app**
   ```
   streamlit run app.py
   ```

## Usage
   - Enter News Article URLs: In the sidebar, input up to three news article URLs.
   - Process the URLs: Click the "Process URLs" button to start loading and processing the content from the specified URLs.
   - Ask a Question: Once the data is processed, you can enter any question related to the articles in the "Question" input field.
   - View Results: The application will display the answer to your question along with sourced references.

## Example

- Enter the URLs of relevant news articles in the sidebar.
- Click on "Process URLs" to extract and analyze the content.
- Type your question, for example, "What are the main points discussed in the articles?" and hit Enter.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
This application utilizes LangChain and Groq for handling language models and document processing.
Special thanks to the open-source community for providing powerful libraries and frameworks.
