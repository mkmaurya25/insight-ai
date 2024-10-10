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
   git clone https://github.com/yourusername/insightai-news-research.git
   cd insightai-news-research

2. **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

Create a .env file in the root directory of the project and add your environment variables, such as API keys, if necessary.
    ```
    # Example .env content
    YOUR_API_KEY=your_api_key_here
    ```