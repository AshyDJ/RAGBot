
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain_core.vectorstores import InMemoryVectorStore
import json
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

load_dotenv()

from flask import *
from flask_cors import CORS, cross_origin
app = Flask(__name__) 
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

previous_split_document = None


@app.route("/session/<file_name>/load_session_chats",methods=["GET"])
def load_session_chats_ai(file_name):
    with open("Session_history.json","r") as read_json_file:
        json_data=json.load(read_json_file)

    session_chats=json_data.get(file_name)

    return jsonify({"session_chats":session_chats}),200


@app.route("/load_session_documents",methods=["GET"])
def load_session_documents_ai():
    session_documents = [f for f in os.listdir(os.path.join(os.getcwd(), "Documents"))]
    return jsonify({"session_documents":session_documents}),200


@app.route("/create_session", methods=["POST"])
def create_session_ai():
    global previous_split_document
    file = request.files["file"]
    file.save(os.path.join(os.path.join(os.getcwd(), "Documents"), file.filename))

    loader=PyPDFLoader(os.path.join(os.path.join(os.getcwd(), "Documents"), file.filename))
    docs = loader.load()

    
    current_split_document = text_splitter.split_documents(docs)
    safe_collection_name = clean_collection_name(file.filename)

    persist_dir = os.path.join("chroma_sessions", safe_collection_name)
    vector_store = Chroma(
        collection_name=safe_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vector_store.add_documents(current_split_document)
    

    with open("Session_history.json","r") as read_json_file:
        json_data=json.load(read_json_file)

    json_data.update({file.filename:[]})

    with open("Session_history.json", "w") as write_json_file:
        json.dump(json_data, write_json_file, indent=4)

    return "",200


@app.route("/session/<file_name>", methods=["POST"])
def prompt_ai(file_name):
    data = request.get_json()
    prompt = data.get("prompt")

    with open("Session_history.json","r") as read_json_file:
        json_data=json.load(read_json_file)

    chat_memory=json_data.get(file_name)
    chat_memory, condensed_context = summarize_if_needed(chat_memory)

    



    safe_collection_name = clean_collection_name(file_name)

    persist_dir = os.path.join("chroma_sessions", safe_collection_name)
    vector_store = Chroma(
        collection_name=safe_collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    
    retrieved_docs = vector_store.similarity_search(prompt)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    full_prompt = f"Chat history:\n{condensed_context}\n\nRelevant document content:\n{docs_content}\n\nAnswer the question: {prompt}"

    response = llm.invoke(full_prompt).content
    print(response)

    
    chat_memory.append(prompt)
    chat_memory.append(response)
    json_data.update({file_name:chat_memory})


    with open("Session_history.json", "w") as write_json_file:
        json.dump(json_data, write_json_file, indent=4)

    return jsonify({"response": response,"chat_memory":chat_memory}),200

@app.route("/") 
def index(): 
    return "Backend is online"




'''
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
'''

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    max_tokens=300,
    timeout=100,
    max_retries=2,
    # other params...
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)



def summarize_if_needed(chat_memory):
    if len(chat_memory) <= 8:
        return chat_memory, "\n".join(chat_memory)  # No summarization needed

    # Separate old and recent memory
    old_turns = chat_memory[:len(chat_memory) - 4]
    recent_turns = chat_memory[-4:]

    # Summarize the early part
    summary_prompt = "Summarize the following conversation:\n" + "\n".join(old_turns)
    summary_response = llm.invoke(summary_prompt)

    # Compose condensed context (used for prompt)
    summarized_context = f"Earlier conversation:\n{summary_response.content}\n\nRecent messages:\n" + "\n".join(recent_turns)

    return chat_memory, summarized_context



def clean_collection_name(filename: str) -> str:
    # Remove extension and replace spaces/special chars with underscores
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    # Strip leading/trailing invalid chars (_ . -)
    name = name.strip("_.-")
    return name



app.run(debug=True) 



'''
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_memory: List[str]


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs,"chat_memory":state["chat_memory"]}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    #if state["chat_memory"]:
    chat_memory = "\n".join(state["chat_memory"])  
    full_context = f"Memory of the Prompter's previous Session with you(the model):\n{chat_memory}\n\nContext from documents:\n{docs_content}"
    #else:
    #    full_context = f"\n\nContext from documents:\n{docs_content}"
    
    messages = prompt.invoke({"question": state["question"], "context": full_context})
    response = llm.invoke(messages)
    return {"answer": response.content}




# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
'''

