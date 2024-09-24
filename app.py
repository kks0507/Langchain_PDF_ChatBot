from flask import Flask, request, jsonify, render_template
from langchain_utils import LangChainHelper #Langchain_utils.py에서 구현된 클래스이며, PDF 파일을 처리하고 질문에 대한 답변을 생성하는 역할
import os

# Flask 애플리케이션을 정의하며, 웹 서버의 역할을 수행
app = Flask(__name__)

# 업로드 폴더 생성 확인
# PDF 파일을 업로드하기 전에 uploads 폴더가 존재하는지 확인하고, 존재하지 않으면 폴더를 생성
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# LangChainHelper 인스턴스 생성 (클래스를 초기화하여 lc_helper 객체를 생성, 이 객체는 PDF 파일을 처리하고 질문에 대한 답변을 생성에 활용) 
lc_helper = LangChainHelper()

@app.route('/')
def index():
    return render_template('langchain.html')

#  PDF 파일 업로드 API (사용자가 PDF 파일을 업로드하면 이 API가 호출)
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "파일을 선택해주세요."}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    lc_helper.process_pdf(pdf_file)   # LangChainHelper를 사용하여 업로드된 PDF 파일을 처리. PDF 파일을 텍스트로 변환하고 벡터로 임베딩하여 벡터 스토어에 저장
    return jsonify({"message": "PDF 파일이 성공적으로 업로드되었습니다."})

# 질문을 통해 답변을 받는 API (사용자가 질문을 입력하면 해당 API가 호출)
@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question') # POST 요청으로 전달된 JSON 데이터에서 question 키의 값을 가지고 온다.
    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    answer = lc_helper.get_answer(question) #LangChainHelper에서 질문에 대해 생성한 답변을 가지고 온다.
    return jsonify({"answer": answer})

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(debug=True)  # Flask 애플리케이션을 로컬에서 실행할 때, debug=True로 설정하여 개발 모드에서 실행. 이 모드는 오류 발생 시 자세한 정보를 콘솔에 출력


