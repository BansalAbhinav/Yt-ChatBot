# 🎬 YouTube Transcript GPT Assistant

A **Streamlit-based RAG (Retrieval-Augmented Generation) chatbot** that allows users to ask questions about YouTube video transcripts. The chatbot extracts the transcript of a YouTube video, splits it into chunks, generates embeddings, and uses a Google Generative AI model to answer questions **strictly based on the transcript**.

---


## Live Link

```bash
https://ytchatbotbansal.streamlit.app/
```

## Features

- Fetches **YouTube video transcripts** in English.
- Splits transcripts into chunks for better retrieval.
- Uses **Google Generative AI** with embeddings to answer questions.
- Retrieval-Augmented Generation (RAG) ensures answers are based **only on the video transcript**.
- Handles errors such as unavailable videos or missing transcripts.
- Professional Streamlit UI with:
  - Video preview
  - Question input box
  - Display of concise answers

---

## Demo

1. Enter a YouTube video URL.
2. Click **Load Video** to fetch and process the transcript.
3. Ask questions about the video using the input box.
4. Get concise, transcript-based answers instantly.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/BansalAbhinav/Yt-ChatBot.git
cd Yt-ChatBot


2.Create a virtual environment and install dependencies:
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt

```
3. Set up environment variables by creating a .env file:

```bash
 GOOGLE_API_KEY=your_google_api_key_here
```

4.License
```bash
This project is licensed under the MIT License.
```