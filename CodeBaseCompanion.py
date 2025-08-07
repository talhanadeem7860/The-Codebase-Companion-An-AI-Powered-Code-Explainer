import os
import warnings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


warnings.filterwarnings("ignore", category=DeprecationWarning)

def create_vector_db_from_codebase(repo_path):
    """
    Loads a code repository, splits it into language-aware chunks,
    creates embeddings, and stores them in a FAISS vector store.
    """
    db_file_path = f"faiss_index_{os.path.basename(repo_path)}"

    # Check if the vector store already exists
    if os.path.exists(db_file_path):
        print(f"[INFO] Loading existing vector store from {db_file_path}...")
        return FAISS.load_local(db_file_path, HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

    print(f"[INFO] Creating new vector store for codebase at {repo_path}...")
    # Load only Python files from the directory
    loader = DirectoryLoader(repo_path, glob="**/*.py", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    # Split the documents into chunks using a Python-specific splitter
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language="python", chunk_size=1000, chunk_overlap=150
    )
    docs = python_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create the FAISS vector store and save it
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_file_path)
    print(f"[INFO] Vector store created and saved to {db_file_path}.")
    return db

def create_qa_chain(db, llm):
    """
    Creates a retrieval-based Q&A chain for code explanation.
    """
    prompt_template = """
    You are an expert programmer and a helpful coding assistant.
    Use the following pieces of code context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know.
    Don't try to make up an answer. Provide a clear, concise, and high-level explanation.

    Context: {context}

    Question: {question}
    
    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Application Logic ---
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this application.")
    else:
        repo_path = input("Enter the full path to your local code repository: ").strip()
        
        if not os.path.isdir(repo_path):
            print(f"[ERROR] The path '{repo_path}' is not a valid directory.")
        else:
            # 1. Create or load the vector store
            vector_store = create_vector_db_from_codebase(repo_path)
            
            if vector_store:
                # 2. Create the LLM and Q&A chain
                llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
                qa_chain = create_qa_chain(vector_store, llm)
                
                # 3. Start interactive Q&A session
                print("\n[INFO] Codebase processed. You can now ask questions about it.")
                print("Type 'exit' to quit.")
                
                while True:
                    user_question = input("\nYour Question: ")
                    if user_question.lower() == 'exit':
                        break
                    
                    # Get the answer from the chain
                    response = qa_chain({"query": user_question})
                    
                    # Print the answer
                    print("\nAnswer:")
                    print(response["result"])
                    
                    # Print the sources used
                    print("\nSources (Relevant Code Snippets):")
                    for i, source in enumerate(response["source_documents"]):
                        # source.metadata['source'] contains the file path
                        print(f"  Source {i+1}: {os.path.basename(source.metadata['source'])}")
                        print(f"  ```python\n{source.page_content[:200].strip()}...\n  ```")