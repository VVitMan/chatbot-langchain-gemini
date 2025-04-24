import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load .env file
load_dotenv()
# Get API key from environment
google_api_key = os.getenv("GOOGLE_API_KEY")
# Configure genai
genai.configure(api_key=google_api_key)

# Helper function to clean text by removing invalid surrogates
def clean_text(text):
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")

# Read all PDF files and return cleaned text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw = page.extract_text()
            if raw:
                cleaned = clean_text(raw)
                text += cleaned
    if not text.strip(): #
        raise ValueError("No valid text found in uploaded PDF files.")
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Embeddings for each chunk
def get_vector_store(chunks):
    if not chunks:
        raise ValueError("No text chunks found. Ensure the PDF contains readable text.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
# Conversational
def get_conversational_chain():
    prompt_template = """
    You are an academic assistant designed to provide detailed, accurate, and formal responses based strictly on the given context. If the answer is not found in the provided context, respond with: "The requested information is not available within the supplied context."
    In addition to answering the user's question, if applicable, briefly suggest relevant or related academic research areas, concepts, or studies that may enhance the user's understanding.
    After the answer, if applicable, briefly give key points of the context provided.
    User may referring to the provided context as "PDF", "PDFs", "document", "documents","text", or something around that.

    Please response in the same language as the question. except the user asks for a specific language, then respond in that language.
    User may write very curt questions, like 'summary', 'date', 'สรุป' please assume that the user is talking about the provided contents.

    If the answer is not in provided context just say, "The requested information is not available within the supplied context. However, you may find it useful to explore related studies on [].", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Handle User Input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    return response['output_text']

# Clear Chat
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

# Streamlit app
def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🖐",
        layout="wide"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                non_pdf_files = [file.name for file in pdf_docs if not file.name.lower().endswith(".pdf")]
                
                if non_pdf_files:
                    st.error(f"Only PDF files are allowed. The following are not PDFs: {', '.join(non_pdf_files)}")
                else:
                    with st.spinner("Processing..."):
                        try:
                            raw_text = get_pdf_text(pdf_docs)
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("Success! Ready to chat.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            st.error("Please upload at least one PDF file.")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files using Gemini 🙋‍♂️")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    
    # Chat input
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"**User:** {prompt}")

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = user_input(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(f"**Assistant:** {response}")
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        st.markdown(f"**Assistant:** {error_message}")

if __name__ == "__main__":
    main()
