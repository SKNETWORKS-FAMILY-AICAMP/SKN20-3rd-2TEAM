import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 경로 설정
SRC_DIR = Path(__file__).parent  # 04_ui 폴더
PROJECT_ROOT = SRC_DIR.parent.parent  # SKN20-3rd-2TEAM 폴더

# 필요한 모듈 경로 추가 및 모듈 임포트
sys.path.insert(0, str(SRC_DIR.parent / "02_utils"))
from vectordb import load_vectordb

sys.path.insert(0, str(SRC_DIR.parent / "03_rag"))
from simpleRAGsystem_5 import SimpleRAGSystem

# Flask 앱 생성
app = Flask(__name__, static_folder=str(SRC_DIR))
CORS(app)

# RAG 시스템 초기화
MODEL_NAME = os.getenv("MODEL_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

print("🔄 VectorDB 로딩 중...")
vectorstore = load_vectordb(
    model_name=MODEL_NAME,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
print("✅ VectorDB 로딩 완료!")

print("🔄 LLM 초기화 중...")
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
print("✅ LLM 초기화 완료!")

print("🔄 RAG 시스템 초기화 중...")
rag_system = SimpleRAGSystem(vectorstore, llm)
print("✅ RAG 시스템 초기화 완료!")


@app.route('/')
def index():
    """메인 HTML 페이지 제공"""
    try:
        return send_from_directory(str(SRC_DIR), 'index.html')
    except Exception as e:
        print(f"❌ HTML 파일을 찾을 수 없습니다: {str(e)}")
        return f"index.html 파일을 찾을 수 없습니다. 경로: {SRC_DIR}/index.html", 404


@app.route('/api/chat', methods=['POST'])
def chat():
    """채팅 API 엔드포인트 (chat 메서드 사용)"""
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': '메시지가 비어있습니다.'}), 400
        
        print(f"📩 받은 질문: {user_message}")
        
        # RAG 시스템의 chat 메서드 사용 (히스토리 저장됨)
        response = rag_system.chat(user_message)
        
        print(f"✅ 답변 생성 완료")
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ask', methods=['POST'])
def ask():
    """질문 답변 API (출처 포함, ask_with_sources 메서드 사용)"""
    try:
        data = request.json
        question = data.get('question', '')
        score_threshold = data.get('score_threshold', 0.7)
        
        if not question:
            return jsonify({'error': '질문이 비어있습니다.'}), 400
        
        print(f"📩 받은 질문: {question}")
        
        # RAG 시스템의 ask_with_sources 메서드 사용
        result = rag_system.ask_with_sources(
            question=question,
            stream=False,
            score_threshold=score_threshold
        )
        
        print(f"✅ 답변 생성 완료")
        print(f"📚 출처 개수: {len(result['sources'])}")
        
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'sources': result['sources']
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """대화 히스토리 초기화"""
    try:
        rag_system.clear_history()
        print("🔄 대화 히스토리 초기화됨")
        
        return jsonify({
            'success': True,
            'message': '대화 히스토리가 초기화되었습니다.'
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """통계 정보 반환"""
    try:
        return jsonify({
            'success': True,
            'stats': {
                'total_papers': 506,
                'total_keywords': 1449,
                'chat_history_length': len(rag_system.chat_history)
            }
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask 서버 시작...")
    print("📍 http://localhost:5000 에서 접속 가능합니다.")
    print(f"📁 HTML 파일 위치: {SRC_DIR}/index.html")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)