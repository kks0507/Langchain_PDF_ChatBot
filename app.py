from flask import Flask, request, jsonify, render_template
from langchain_utils import LangChainHelper
import os

app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# LangChainHelper instance
lc_helper = LangChainHelper()

@app.route('/')
def index():
    return render_template('langchain.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "파일을 선택해주세요."}), 400

    pdf_file = request.files['pdf']
    if pdf_file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    lc_helper.process_pdf(pdf_file)
    return jsonify({"message": "PDF 파일이 성공적으로 업로드되었습니다."})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "질문을 입력해주세요."}), 400

    answer = lc_helper.get_answer(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)


