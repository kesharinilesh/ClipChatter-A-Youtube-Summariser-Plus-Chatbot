from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import YoutubeLoader
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import gradio as gr
import os
import time
import shutil


def process_and_transcribe(url, openai_api_key, model):
    """Process and transcribe the video at a given url"""
    # Setting qa_chain as a global variable
    global qa_chain
    # save_dir = some_title
    # loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParser(api_key = openai_api_key))
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    vectordb = FAISS.from_texts(splits, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=model, temperature=0, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return text

def response(message, history):
    return qa_chain.run(message)

transcribe_interface = gr.Interface(
    fn=process_and_transcribe, 
    inputs=['text', 'text', gr.components.Radio(['gpt-3.5-turbo', 'gpt-3.5-turbo-16k','gpt-4'])],
    outputs=['text'],
    title="Clip Chatter",
    description=""" Welcome to the ClipChatter: Your Video Summarizer & Follow-Up Chatbot!\n
    \n1. Paste the link to your desired YouTube video in the box below.
    \n2. Share the secret passphrase (your OpenAI API Key) to unlock the power of conversation with this AI model.\n
    \n3. Select your preferred Model:
        - GPT-3.5-turbo for quick summaries,
        - GPT-3.5-turbo-16k for deeper insights,
        - GPT-4 for the ultimate brainstorming session.
    \n4. Once the transcription is complete, hop over to the Chat interface
    \n5. Type 'Summarize' or discuss anything related to the video, this model is here to assist you!
    \n6. Want to clear the slate? Hit the Clear button and start a new!
    So, Sit back and relax!"""
)

chat_interface = gr.ChatInterface(fn=response, title="Chat", description="Let's dive into the world of knowledge and banter! \nFeel free to Ask or discuss anything related to the video. Our model is here to assist you!")
demo = gr.TabbedInterface([transcribe_interface, chat_interface], ["Video Transcriber", "ChatterBot"])
demo.queue()
demo.launch()