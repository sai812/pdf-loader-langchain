from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, request, jsonify
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize in-memory vector store
vectorstore = None
conversation_chain = None

def read_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def initialize_vectorstore(text_chunks):
    global vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def initialize_conversation_chain():
    global conversation_chain
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    pdfs = request.files.getlist('file')
    pdf_text = read_pdf(pdfs)
    text_chunks = get_text_chunks(pdf_text)
    initialize_vectorstore(text_chunks)
    initialize_conversation_chain()
    
    return jsonify({'message': 'File uploaded and text processed successfully'})

@app.route('/query', methods=['POST'])
def query():
    global conversation_chain
    data = request.json
    user_question = data.get('question')

    if not user_question:
        return jsonify({'error': 'Question is required'}), 400

    if conversation_chain is None:
        return jsonify({'error': 'No conversation chain available. Upload a PDF first.'}), 400

    try:
        response = conversation_chain({'question': user_question})
        return jsonify({'response': response['answer']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
