import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

class LangChainHelper:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chat_history = []  # 대화 기록 저장을 위한 리스트

    def process_pdf(self, pdf_file):
        """Process the uploaded PDF file."""
        file_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(file_path)
        
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        data = loader.load()

        # Extract and split text
        text = "".join([doc.page_content for doc in data])
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Embed the text and store in Chroma DB
        self.vector_store = Chroma.from_texts(chunks, self.embeddings)

    def get_answer(self, question):
        """Search for similar documents and generate an answer."""
        if self.vector_store is None:
            return "먼저 PDF 파일을 업로드하세요."

        # Search for similar documents
        docs = self.vector_store.similarity_search(question)

        # Generate answer using GPT model
        llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0.1)
        chain = ConversationalRetrievalChain.from_llm(llm, self.vector_store.as_retriever())

        # 대화 기록을 함께 전달하여 답변 생성
        # 대화 기록을 함께 전달하여 답변 생성
        inputs = {
            "question": question,
            "chat_history": self.chat_history
        }

        # 대화 기록 업데이트

        result = chain.run(inputs)

        # 대화 기록 업데이트
        self.chat_history.append((question, result))

        return result
