from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask,render_template, request

app = Flask(__name__)
def read_pdf(pdfs):
    text= ""
    for pdf in pdfs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
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


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def conv(conversation):
    user_que="india capital"
    response = conversation({'question':user_que})
    print(response)

@app.route('/upload',methods=['POST']) 
def upload():
    load_dotenv()
    print('test')
    #pdf read
    pdfs=request.files.getlist('file')
    pdf_text=read_pdf(pdfs)
    text_chunks = get_text_chunks(pdf_text)
    print(text_chunks)
    vectorstore=get_vectorstore(text_chunks)
    print('t2',vectorstore)
    conversation = get_conversation_chain(vectorstore) 
    response=conv(conversation)
    return response



if __name__ == '__main__':
    app.run(debug=True)
    # print('hello')