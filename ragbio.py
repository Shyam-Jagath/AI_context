import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Load PDF

def bio():
    pdf_path = "kebo1ps_merged.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(f"Failed to load HuggingFace embeddings: {e}")

  
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

   
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(model="llama3-70b-8192"),  # You can try mixtral/gemma too
            retriever=retriever,
            return_source_documents=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize QA chain: {e}")

   
    print("PDF-based RAG Agent. Ask me anything (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = qa_chain.run(user_input)
            print("AI:", response)
        except Exception as e:
            print(" Error during response generation:", e)
