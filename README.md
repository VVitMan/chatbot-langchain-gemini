# Chatbot Gemini PDF

---

## Feature
- 📄 Upload and process multiple PDFs
- 🤖 Ask questions in natural language
- 🧠 Intelligent answers using Google Gemini
- 🧷 Text chunking and semantic search with FAISS
- 🔐 API key management with `.env`
- 🖥️ Streamlit-powered UI for an interactive experience

---

## 🛠 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

---

## 🔧 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/VVitMan/chatbot-langchain-gemini.git
cd chatbot-langchain-gemini
```

### 2. Create & Activate a Virtual Environment (Optional but Recommended)
For Windows
```
python -m venv venv
venv\Scripts\activate
```
For macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Set up the .env file
Create a **.env** file in the root directory and add your Google Generative AI API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```
Don't have a key? Get one from | [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Run app
```
streamlit run app.py
```
Then open the app in your browser: http://localhost:8501

