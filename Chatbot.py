import streamlit as st
import time

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

import streamlit as st

st.set_page_config(page_title="Home Page - Chatbot",)

st.write("# Welcome to Streamlit! 👋")
st.write("**👈 Select a tab from the sidebar** to diagnose Alzheimer's disease.")
st.write("Or you can have a conversation with our chatbot right below. 👇")

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
    "My name is Ron, your AI assistant. I can provide you some information about Alzheimer's disease.",
    "Who are you?",
    "My name is Ron, your AI assistant. I can provide you some information about Alzheimer's disease.",
    "Can you tell me your name?",
    "My name is Ron, your AI assistant.",
    "Introduce yourself",
    "My name is Ron, your AI assistant. I can provide you some information about Alzheimer's disease.",

    "How are you?",
    "I'm good. How can I assist you?",

    "What can you do?",
    "I can provide you some information about Alzheimer's disease. Please note that I'm still in developing, so I may not able to understand all your questions. But I will try my best to help you!\n\n"
    "If you want to diagnose Alzheimer's disease, please **👈 check out the tabs from the sidebar**.",

    "I want to diagnose Alzheimer's disease.",
    "If you want to diagnose Alzheimer's disease, please **check out the tabs from the sidebar**.",
    "Help me to diagnosis Alzheimer's disease",
    "👈 Please **check out the tabs from the sidebar**, you can diagnose Alzheimer's disease by uploading MRI brain images or entering data there.",

    "What is Alzheimer's disease?",
    "Alzheimer's disease is a neurodegenerative disease that usually starts slowly and progressively worsens. It is the cause of 60–70% of cases of dementia. "
    "Early symptoms of Alzheimer's disease include forgetting recent events or conversations. Over time, Alzheimer's disease leads to serious memory loss and affects a person's ability to do everyday tasks. "
    "There is no cure for Alzheimer's disease. In advanced stages, loss of brain function can cause dehydration, poor nutrition or infection. These complications can result in death. But medicines may improve symptoms or slow the decline in thinking.",
    
    "Tell me about Alzheimer's disease.",
    "Alzheimer's disease is a neurodegenerative disease that usually starts slowly and progressively worsens. It is the cause of 60–70% of cases of dementia. "
    "Early symptoms of Alzheimer's disease include forgetting recent events or conversations. Over time, Alzheimer's disease leads to serious memory loss and affects a person's ability to do everyday tasks. "
    "There is no cure for Alzheimer's disease. In advanced stages, loss of brain function can cause dehydration, poor nutrition or infection. These complications can result in death. But medicines may improve symptoms or slow the decline in thinking.",

    "What cause of Alzheimer's disease?",
    "The causes of Alzheimer's disease remain poorly understood. There are many environmental and genetic risk factors associated with its development. The strongest genetic risk factor is from an allele of apolipoprotein E. "
    "Other risk factors include a history of head injury, clinical depression, and high blood pressure. The progression of the disease is largely characterised by the accumulation of malformed protein deposits in the cerebral cortex, called amyloid plaques and neurofibrillary tangles. "
    "These misfolded protein aggregates interfere with normal cell function, and over time lead to irreversible degeneration of neurons and loss of synaptic connections in the brain. "
    "A probable diagnosis is based on the history of the illness and cognitive testing, with medical imaging and blood tests to rule out other possible causes."
    "Initial symptoms are often mistaken for normal brain aging. Examination of brain tissue is needed for a definite diagnosis, but this can only take place after death.",

    "Can Alzheimer's disease be cured?",
    "No treatments can stop or reverse its progression, though some may temporarily improve symptoms. A healthy diet, physical activity, and social engagement are generally beneficial in aging, and may help in reducing the risk of cognitive decline and Alzheimer's.",
    "Are there any treatments for Alzheimer's disease?",
    "No treatments can stop or reverse its progression, though some may temporarily improve symptoms. A healthy diet, physical activity, and social engagement are generally beneficial in aging, and may help in reducing the risk of cognitive decline and Alzheimer's.",

    "What are the symptoms of Alzheimer's disease",
    "Alzheimer's disease, a progressive form of dementia, manifests with symptoms like **memory loss**, **difficulty with language**, and **changes in behavior and personality**, often starting subtly and worsening over time. "
    "At first, someone with the disease may be aware of having trouble remembering things and thinking clearly. Over time, Alzheimer's disease leads to serious memory loss and affects a person's ability to do everyday tasks."
    "Alzheimer's disease causes trouble concentrating and thinking, especially about abstract concepts such as numbers. Doing more than one task at once is especially hard. "
    "Routine activities that involve completing steps in a certain order also can be hard for people with Alzheimer's disease. As Alzheimer's disease becomes advanced, people forget how to do basic tasks such as dressing and bathing.",

    "What are the stages of Alzheimer's disease?",
    "Alzheimer’s disease has three stages: **mild, moderate and advanced**. They’re not exactly the same for everyone, and they may happen more slowly or quickly for different people.",
    "How many stages do Alzheimer's disease have?",
    "Alzheimer’s disease has three stages: **mild, moderate and advanced**. They’re not exactly the same for everyone, and they may happen more slowly or quickly for different people.",

    "How long do Alzheimer's patients live?",
    "On average, people with Alzheimer's disease live **between three and 11 years** after diagnosis. "
    "But some live 20 years or more. The degree of impairment at diagnosis can affect life expectancy. Untreated vascular risk factors such as hypertension are associated with a faster rate of progression of Alzheimer's disease.", 
    "How long can people with Alzheimer's live?",
    "On average, people with Alzheimer's disease live **between three and 11 years** after diagnosis. "
    "But some live 20 years or more. The degree of impairment at diagnosis can affect life expectancy. Untreated vascular risk factors such as hypertension are associated with a faster rate of progression of Alzheimer's disease.", 

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

st.title("💬 Chat with ChatterBot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything..."):
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