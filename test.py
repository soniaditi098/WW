import streamlit as st
import requests
#from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
#import csv
# from datetime import datetime
# import time  # Add this import at the beginning of your script
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
#import os

# load_dotenv(dotenv_path="ai.env")

openai_api_key = os.getenv("OPENAI_API_KEY")

import os
import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def get_huggingface_token():
    try:
        with open("./hf_token.txt", "r") as file:
            hf_token = file.readline().strip()
        return True, hf_token
    except FileNotFoundError:
        return False, ""
        
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def initialize_huggingface_components(filename, persistent_directory='./chroma_db'):
    success, hf_token = get_huggingface_token()

    if success:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # Read data from file and split into paragraphs
        loader = TextLoader(filename)
        data = loader.load()

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "(?<=\. )", " "]
        )
        paras = r_splitter.split_documents(data)

        # Initialize Hugging Face embeddings
        embeddings_model = HuggingFaceEmbeddings()

        # Create vector store
        vectordb = Chroma.from_documents(
            documents=paras,
            embedding=embeddings_model,
            persist_directory=persistent_directory,
        )

        return vectordb
    
def get_openai_key_and_answer(query, vectordb):
    # success, open_ai_key = get_openai_key()

    # if success:
    #     llm = OpenAI(openai_api_key=open_ai_key)
    if openai_api_key:
        llm = OpenAI(openai_api_key=openai_api_key)

        new_line = '\n'
        template = f"Use the following pieces of context to answer truthfully.{new_line}If the context does not provide the truthful answer, make the answer as truthful as possible.{new_line}Use 15 words maximum. Keep the response as concise as possible.{new_line}{{context}}{new_line}Question: {{question}}{new_line}Response: "
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        # Run question-answering chain
        qa_chain = RetrievalQA.from_chain_type(llm,
                                               retriever=vectordb.as_retriever(),
                                               return_source_documents=True,
                                               chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        result = qa_chain({"query": query})
        return result["result"]

    else:
        print("Error: OpenAI key not found. Please check your file.")
        return None

def get_openai_key():
    try:
        with open("./open-ai-key.txt", "r") as file:
            open_ai_key = file.readline().strip()
        success = True
    except FileNotFoundError:
        open_ai_key = ""
        success = False

    return success, open_ai_key

# tokenizer = AutoTokenizer.from_pretrained("AyoubChLin/DistilBERT_ZeroShot")
# model = AutoModelForSequenceClassification.from_pretrained("AyoubChLin/DistilBERT_ZeroShot")

# classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
topics = ["Health", "Environment", "Technology", "Economy", "Entertainment", "Sports", "Politics", "Education", "Travel", "Food"]

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import re

# Download NLTK stopwords dataset
import nltk
nltk.download('stopwords')
nltk.download('punkt')

chat_history = []


def preprocess_query(user_query):
    user_query = re.sub(r'[^a-zA-Z0-9\s]', '', user_query)
    tokens = word_tokenize(user_query)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    spell = SpellChecker()
    corrected_tokens = [spell.correction(word) for word in filtered_tokens]
    corrected_tokens = [word.replace('gud', 'good') for word in corrected_tokens]
    preprocessed_query = ' '.join(corrected_tokens)

    return preprocessed_query

    
def search_solr(query, base_url, return_fields, num_results=10):
 
    q = f'{query}'  # Modify this as needed
    qf = 'title^2.0'  # Example: Boost the title field with a weight of 2.0
    pf = 'title^2.0'  # Example: Boost the title field with a weight of 2.0
    search_url = f'{base_url}/select?q={q}&qf={qf}&pf={pf}&fl={return_fields}&wt=json&sort=score desc&rows={num_results}&defType=edismax'
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f'Failed to fetch data: Status Code {response.status_code}'}
    
def search_solr_by_topic(topic, base_url, return_fields, num_results=10):
    fq = f'topic:("{topic}")'
    search_url = f'{base_url}/select?q=*:*&fq={fq}&fl={return_fields}&wt=json&sort=score desc&rows={num_results}'
    response = requests.get(search_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': f'Failed to fetch data: Status Code {response.status_code}'}
    
def classify_topic(user_query):
    result = classifier(user_query, topics)
    return result["labels"][0]

def classify_end_continue(prompt):
    url = "http://34.127.56.91:5000/classify_end_continue"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

def classify_wiki_chat(prompt):
    url = "http://34.168.204.151:5000/classify_wiki_chat"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json()

def chatterbot_response(prompt):
    url = "http://34.168.245.140:5000/chat"
    data = {"prompt": prompt}
    response = requests.post(url, json=data)
    return response.json().get('output', 'Sorry, I could not understand that.')


def write_to_file(data, filename="data.txt"):
    summaries_list = [entry['summary'][0] for entry in data]
    data = '\n'.join(summaries_list)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)
    return filename
        
def on_send():
    user_input = st.session_state.user_input.strip()
    if user_input:
        wiki_chat_result = None
        if st.session_state.get('conversation_ended', False):
            st.session_state.user_input = ''
            return
        end_continue_response = classify_end_continue(user_input)
        if end_continue_response['output'] == 'continue chat':
            end_continue_result='continue chat'
            wiki_chat_response = classify_wiki_chat(user_input)

            if wiki_chat_response['output'] == 'wiki':
                wiki_chat_result='wiki'
                preprocessed_query = preprocess_query(user_input)
                results = search_solr(preprocessed_query, 'http://35.245.97.133:8983/solr/IRF23P1', 'topic,title,revision_id,summary')
                if 'response' in results and 'docs' in results['response'] and len(results['response']['docs']) > 0:
                    latest_doc = results['response']['docs'][0]
                    #response = latest_doc.get('summary', 'Summary not available.')
                    data = results['response']['docs']
                    resulting_file = write_to_file(data)
                    vectordb=initialize_huggingface_components(resulting_file)
                    response=get_openai_key_and_answer(user_input, vectordb)
                    #response = latest_doc.get('summary', 'Summary not available.')
                    #update_chat_history(user_input, response)  
                    update_chat_history(user_input, response, end_continue_result, wiki_chat_result)


                else:
                    response = "I couldn't find any information on that topic."
                    update_chat_history(user_input, response, end_continue_result, wiki_chat_result)


            elif wiki_chat_response['output'] == 'chat':
                wiki_chat_result='chat'
                response = chatterbot_response(user_input)
                #update_chat_history(user_input, response)
                update_chat_history(user_input, response, end_continue_result, wiki_chat_result)


        elif end_continue_response['output'] == 'bye':
            end_continue_result= 'bye'
            response = chatterbot_response(user_input)
            update_chat_history(user_input, response, end_continue_result, wiki_chat_result)
            st.session_state['conversation_ended'] = True

#         else:
#             response = "I'm not sure how to respond to that."
#             update_chat_history(user_input, response) 
        
        
        # log_chat_to_csv(
        #     user_input, 
        #     response, 
        #     end_continue_result, 
        #     wiki_chat_result
        # )


        #update_chat_history(user_input, response)
        st.session_state.user_input = ''  


def log_chat_to_csv(user_input, bot_response, end_continue_result, wiki_chat_result):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open('chat_history.csv', 'a', newline='', encoding='utf-8') as file:
        fieldnames = [
            "user_input", 
            "bot_response", 
            "end_continue_result", 
            "wiki_chat_result", 
            "timestamp"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Check if the file is empty to decide whether to write headers
        file.seek(0, 2)
        if file.tell() == 0:
            writer.writeheader()

        # Write the chat data along with classifier results
        writer.writerow({
            "user_input": user_input,
            "bot_response": bot_response,
            "end_continue_result": str(end_continue_result),
            "wiki_chat_result": str(wiki_chat_result),
            "timestamp": timestamp
        })



def update_chat_history(user_input, response, end_continue_result, wiki_chat_result):
    new_chat_history = st.session_state.chat_history + [('USER', user_input), ('WW', response)]
    st.session_state.chat_history = new_chat_history
    #log_chat_to_csv(user_input, response, end_continue_result, wiki_chat_result)


def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [('WW', 'Hello! How may I help you today?')]
    if 'last_selected_topic' not in st.session_state:
        st.session_state.last_selected_topic = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ''
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False

    st.title('Wisdom-Whisperer')

    # Custom styles for chat bubbles, sidebar, and title
    st.markdown("""
        <style>
            /* General styles */
            h1 { color: black; }

            /* Styles for chat bubbles */
            .message { border-radius: 25px; padding: 10px; margin: 10px 0; border: 1px solid #e6e9ef; position: relative; }
            .user { background-color: #dbf0d4; color: black; }
            .WW { background-color: #f1f0f0; color: black; }

            /* Styles for sidebar */
            .sidebar .sidebar-content { background-color: #f0f0f0; color: black; font-weight: bold; padding-top: 10px; font-size: 25px; }
            .sidebar-heading { color: black; font-weight: bold; font-size: 25px; }
            .sidebar-content a { color: black; font-weight: bold; font-size: 25px; }
        </style>
    """, unsafe_allow_html=True)
    
    if st.session_state['conversation_ended']:
        if st.button('Start a new conversation'):
            st.session_state['conversation_ended'] = False
            st.session_state.chat_history = [('WW', 'Hello! How may I help you today?')]


    with st.sidebar:
        topics_with_placeholder = ["Select a topic..."] + topics
        selected_topic = st.radio("Choose a topic:", topics_with_placeholder, index=0)
        if selected_topic != "Select a topic..." and selected_topic != st.session_state['last_selected_topic']:
            st.session_state.last_selected_topic = selected_topic
            solr_results = search_solr_by_topic(selected_topic, 'http://35.245.97.133:8983/solr/IRF23P1', 'topic,title,revision_id,summary')
            if 'response' in solr_results and 'docs' in solr_results['response'] and len(solr_results['response']['docs']) > 0:
                summary = solr_results['response']['docs'][0].get('summary', 'Summary not available.')
                update_chat_history('WW', f"Here's the summary on the topic: {selected_topic} - {summary}",None, None)
            else:
                update_chat_history('WW', f"Sorry, I couldn't find any information on the topic: {selected_topic}.",None, None)
#             st.session_state['chat_history'].append(('WW', f"Here's the summary on the topic: {selected_topic}."))
#             st.session_state['last_selected_topic'] = selected_topic

    # Display chat history
    for role, message in st.session_state['chat_history']:
        bubble_class = "user" if role == "USER" else "WW"
        st.markdown(f"<div class='message {bubble_class}'>{message}</div>", unsafe_allow_html=True)

    # Text input for user query
    st.text_input("Send a message...", value=st.session_state.user_input, on_change=on_send, key="user_input")

    # Button to send the message
    st.button('➤', on_click=on_send)

if __name__ == '__main__':
    main()
