from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import logging
import os

# Flask 애플리케이션 초기화 및 CORS 설정
app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TF-IDF 벡터 변환기 로드
tfidf_vectorizer_path = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')
try:
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    app.logger.info("TF-IDF 벡터 변환기 로드 완료.")
except Exception as e:
    app.logger.error(f"TF-IDF 벡터 변환기 로드 실패: {e}")

# URL 분류 모델 로드
clf_model_path = os.path.join(os.path.dirname(__file__), 'MNB_TFIDF.pkl')
try:
    clf_model = joblib.load(clf_model_path)
    app.logger.info("URL 분류 모델 로드 완료.")
except Exception as e:
    app.logger.error(f"URL 분류 모델 로드 실패: {e}")
    
    
    
def tokenize_url(url):
    """URL을 토큰화하여 문자열로 반환."""
    tokens = []
    for part in url.split('/'):
        tokens.extend(part.split('.'))
    return ' '.join(tokens)



# URL 예측 함수 정의
def predict_url(url):
    """URL을 분류하여 'good' 또는 'bad'로 반환."""
    try:
        # URL을 문자열로 변환 (명시적으로)
        if not isinstance(url, str):
            app.logger.warning(f"URL 타입이 문자열이 아닙니다: {type(url)}")
            url = str(url)
        
        # URL 토큰화
        tokenized_url = tokenize_url(url)
        
        # URL을 TF-IDF 벡터로 변환
        url_vector = tfidf_vectorizer.transform([tokenized_url])
        app.logger.info(f"URL 벡터: {url_vector.toarray()}")  # 벡터화 결과 로깅
        prediction = clf_model.predict(url_vector)
        return prediction[0]
    
    except Exception as e:
        app.logger.error(f"URL 예측 중 오류 발생: {e}")
        return None



@app.route('/scan', methods=['POST'])
def scan_qr_code():
    """QR 코드에서 추출된 URL을 받아 분류하여 반환."""
    app.logger.info("URL 분류 요청 수신.")

    # 요청에서 URL 추출
    url = request.json.get('url')
    app.logger.info(f"요청받은 URL: {url}")  # 요청받은 URL을 로그에 기록

    if not url:
        app.logger.warning("URL이 제공되지 않았습니다.")
        return jsonify({'status': 'error', 'message': 'URL이 제공되지 않았습니다.'}), 400

    # URL 분류 예측
    prediction = predict_url(url)
    if prediction == 'good':
        app.logger.info(f"URL '{url}'는 안전합니다.")
        return jsonify({'status': 'good', 'message': '이 URL은 안전합니다.', 'url': url})
    elif prediction == 'bad':
        app.logger.warning(f"URL '{url}'는 위험할 수 있습니다.")
        return jsonify({'status': 'bad', 'message': '이 URL은 보안 위험이 있을 수 있습니다.', 'url': url})
    else:
        app.logger.error(f"URL '{url}' 분류 실패. 예측 결과가 없습니다.")
        return jsonify({'status': 'error', 'message': 'URL을 분류할 수 없습니다.', 'url': url}), 500

if __name__ == '__main__':
    app.run(debug=True)
