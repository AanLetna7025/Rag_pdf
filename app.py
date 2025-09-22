import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"PDF not found: {pdf}")
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text
    # for pdf in pdf_docs:
    #     pdf_reader=PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text+=page.extract_text()
    # return text
    

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks[:100]  


def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    PromptTemplate="""
    Answer the following question based only on the provided context.
    If the answer is not in the context, politely state that you cannot find the answer in the provided information.
    Context:\n{context}?\n
    Question:\n{question}?\n

    Answer:
    """

    model=ChatGoogleGenerativeAI(model="Gemini-2.5-flash",temperature=0.3)
    prompt=PromptTemplate(template=PromptTemplate,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_question):
    """Simple function to answer user questions"""
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    
    chain=get_conversational_chain()
    response=chain({"input_documents":docs,"question":user_question},return_only_outputs=True)
    
    #print("Answer:",response["output_text"])

    if "output_text" in response:
        print("Answer:", response["output_text"])
    elif "answer" in response:
        print("Answer:", response["answer"])
    else:
        print("Answer (raw response):", response)

if __name__ == "__main__":
    pdf_files = [r"C:\Users\user\Documents\mainpro\MainProject_Report_final.pdf"]
    print(f"File exists: {os.path.exists(pdf_files[0])}")

    if os.path.exists(pdf_files[0]):
        print("Processing PDF...")
        text = get_pdf_text(pdf_files)
        chunks = get_text_chunks(text)
        get_vector_store(chunks)
        print("PDF processed successfully!")
        
    
        while True:
            question = input("\nAsk a question (or type 'quit'): ")
            if question.lower() == 'quit':
                break
            user_input(question)
    else:
        print("PDF file not found!")

