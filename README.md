# Langchain_PDF_ChatBot 📄🤖

## Description
OpenAI API와 Langchain 라이브러리를 활용하여 첨부한 PDF 파일을 바탕으로 대화를 할 수 있는 챗봇입니다. 사용자가 업로드한 PDF 파일의 내용을 처리하고, 그 내용을 바탕으로 질문에 답변하는 챗봇입니다.

<br>

## Installation and Execution

### Requirements
프로그램을 실행하기 위해 필요한 라이브러리들이 `requirements.txt` 파일에 명시되어 있습니다. 다음 명령어로 필요한 라이브러리들을 설치하세요:

```bash
pip install -r requirements.txt
```

`requirements.txt` 파일에는 다음 라이브러리들이 포함되어 있습니다:
- flask: Flask 웹 프레임워크
- langchain: LangChain 라이브러리
- chromadb: Chroma 벡터 DB
- sentence-transformers: HuggingFace 임베딩 모델을 위한 패키지
- openai: OpenAI API를 위한 패키지
- pypdf: PDF 처리용 패키지
- langchain-community: LangChain 커뮤니티 확장 기능
- python-dotenv: .env 파일로 환경 변수를 로드하기 위한 패키지

<br>

### 실행 방법
1. OpenAI API 키를 `.env` 파일에 저장하세요:
```
OPENAI_API_KEY=your_openai_api_key
```

2. `app.py` 파일을 실행하여 웹 애플리케이션을 시작하세요:
```bash
python app.py
```

3. 웹 브라우저에서 `http://localhost:5000`으로 이동하여 PDF 파일을 업로드하고 질문을 입력하세요.

<br>

### PDF 업로드 및 질문
- PDF 파일을 업로드한 후, 해당 문서의 내용을 바탕으로 질문을 입력하면 챗봇이 답변을 제공합니다.

<br>

## 주요 함수 설명

#### PDF 처리 및 임베딩 생성: `langchain_utils.py`
```python
def process_pdf(self, pdf_file):
    file_path = os.path.join('uploads', pdf_file.filename)
    pdf_file.save(file_path)

    # PDF 파일을 로드하고 텍스트를 추출
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # 텍스트를 추출하고 분할
    text = "".join([doc.page_content for doc in data])
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # 텍스트를 임베딩하여 Chroma DB에 저장
    self.vector_store = Chroma.from_texts(chunks, self.embeddings)
```
이 함수는 사용자가 업로드한 PDF 파일을 로드하여 텍스트를 추출하고, 이를 작은 텍스트 조각으로 분할한 후 임베딩을 생성합니다. 생성된 임베딩은 `Chroma DB`에 저장되어, 추후 질문에 대한 답변을 생성할 때 사용됩니다.

<br>

#### 질문에 대한 답변 생성: `langchain_utils.py`
```python
def get_answer(self, question):
    if self.vector_store is None:
        return "먼저 PDF 파일을 업로드하세요."

    # 질문과 유사한 문서를 검색
    docs = self.vector_store.similarity_search(question)

    # OpenAI GPT 모델을 통해 답변 생성
    llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(llm, self.vector_store.as_retriever())

    inputs = {"question": question, "chat_history": self.chat_history}
    result = chain.run(inputs)

    # 대화 기록 업데이트
    self.chat_history.append((question, result))

    return result
```
이 함수는 사용자의 질문에 대해 유사한 문서를 검색하고, GPT-3.5를 활용하여 질문에 대한 답변을 생성합니다. 검색된 문서를 바탕으로 문맥을 형성하고 대화 기록을 유지하여 일관성 있는 답변을 제공합니다.

<br>

## 웹 인터페이스 설명: `langchain.html`
```html
<form id="pdf-form" enctype="multipart/form-data">
    <input type="file" id="pdf-file" name="pdf" accept="application/pdf">
    <button type="submit">PDF 업로드</button>
</form>

<input type="text" id="messageInput" placeholder="여기에 질문을 입력하세요..." />
<button onclick="sendMessage()">전송</button>
```
이 HTML 파일은 사용자 인터페이스를 제공합니다. PDF 파일을 업로드하고, 질문을 입력할 수 있는 폼이 있으며, 업로드된 파일을 서버로 전송하고 질문에 대한 답변을 반환하는 기능을 포함하고 있습니다.

<br>

## Contributor
- kks0507

## License
This project is licensed under the MIT License.

## Repository
코드 및 프로젝트의 최신 업데이트는 [여기](https://github.com/kks0507/Langchain_PDF_ChatBot.git)에서 확인할 수 있습니다.
