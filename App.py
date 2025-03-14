import streamlit as st
import random
import time

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

import streamlit as st

st.set_page_config(page_title="Home Page",)

st.write("# Welcome to Streamlit! 👋")

# Initialize chatbot
chatbot = ChatBot(
    'Ron', 
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            'default_response': 'I am sorry, but I do not understand.',
            "maximum_similarity_threshold": 0.95
        },
        {
            'import_path': 'chatterbot.logic.SpecificResponseAdapter',
            'input_text': 'MRI brain image',
            'output_text': 'Ok! Please upload your MRI brain image here:'
        },
        'chatterbot.logic.MathematicalEvaluation',
    ],
    read_only=True)

conversation = [
    "hello", 
    "Hi there! How can I assist you?",
    "hi", 
    "Hi there! How can I assist you?",
    "hey",
    "Hi there! How can I assist you?",

    "What is your name?",
    "My name is Ron, your AI assistant.",
    "Who are you?",
    "My name is Ron, your AI assistant.",
    "Can you tell me your name?",
    "My name is Ron, your AI assistant.",
    "Introduce yourself",
    "My name is Ron, your AI assistant. I can assist you with diagnosing Alzheimer's disease based on input files.",

    "How are you?",
    "I'm good. How can I assist you?",

    "What can you do?",
    "I can help you to diagnose Alzheimer's disease based on MRI brain images and excel/csv files. Do you need help with diagnosis?",

    "Yes, help me",
    "Do you want to upload MRI brain images or excel/csv files?",
    "Help me to diagnosis Alzheimer's disease",
    "Do you want to upload MRI brain images or excel/csv files?",

    "Thank you!",
    "You're welcome!",
    "Thanks!",
    "You're welcome!",

    "Who created you?",
    "I was created by an AI enthusiast using Python and ChatterBot.",
]

# Train the chatbot
#trainer = ChatterBotCorpusTrainer(chatbot)
#trainer.train("chatterbot.corpus.english")
trainer = ListTrainer(chatbot)
trainer.train(conversation)

# Streamed response emulator
def response_generator(response):
    for word in str(response).split():
        yield word + " "
        time.sleep(0.05)

st.title("Chatbot with ChatterBot & Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Write something"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = chatbot.get_response(prompt)  
        st.write_stream(response_generator(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})