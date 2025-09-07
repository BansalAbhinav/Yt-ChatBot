import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
import os
import re
st.title("🎬 YouTube Transcript GPT Assistant") 
st.write("Paste a YouTube video URL and ask questions about its transcript!")

video_id = st.text_input("Enter YouTube Video Url:")


def extract_video_id(url_or_id):
    """
    Extracts the YouTube video ID from a full URL or returns the ID as-is.
    """
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    if match:
        return match.group(1)
    return url_or_id 


        
if st.button("Load Video"):
    try:
        ytt_api = YouTubeTranscriptApi()
        video_id_only = extract_video_id(video_id)
        fetched = ytt_api.fetch(video_id_only, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in fetched.to_raw_data())
        st.success("Video loaded successfully!")
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()
    except NoTranscriptFound:
        st.error("No transcript found in the requested language.")
        st.stop()
    except VideoUnavailable:
        st.error("The video is unavailable.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    prompt = PromptTemplate(
        template="""
        You are a highly knowledgeable and careful assistant for a YouTube video. 
Your task is to answer the user's question **using ONLY the provided transcript context**. 
Do not use any outside information. If the answer is not in the transcript, say "I don't know".

Guidelines:
1. Extract the **most relevant information** from the transcript.
2. Provide answers in a **concise and clear manner**.
3. Reason carefully and ensure your answer strictly reflects the transcript.
4. If multiple parts of the transcript are relevant, combine them in your answer.

Transcript Context:
{context}

User Question:
{question}

Answer:
        """,
        input_variables=['context', 'question']
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | model | parser

    st.session_state['main_chain'] = main_chain
    st.session_state['retriever'] = retriever
    st.success("Model initialized! You can now ask questions.")
    
if video_id:
    st.video(video_id, format="video/mp4", start_time=0)


if 'main_chain' in st.session_state:
        
    user_question = st.text_input("Your Question:", key="user_question")
    if st.button("Ask GPT"):
        
        with st.spinner("Generating answer..."):
            answer = st.session_state['main_chain'].invoke(user_question)
        st.write("**Answer:**")
        st.write(answer)
