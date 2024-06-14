# 필요한 라이브러리 설치
!pip install gradio
!pip install transformers
!pip install sentence-transformers
!pip install torch
!pip install keybert

# Gradio 및 Transformers, KeyBERT 라이브러리 불러오기
import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# 텍스트 요약 파이프라인 로드 (영어 요약 모델)
summarizer_english = pipeline('summarization', model='facebook/bart-large-cnn')

def summarize_text(text, language):
    """
    입력 텍스트를 요약하는 함수
    """
    if len(text.split()) < 5:
        return "입력 텍스트가 너무 짧습니다. 더 긴 텍스트를 입력해주세요."
    
    try:
        if language == 'English':
            result = summarizer_english(text, max_length=130, min_length=30, do_sample=False)
            summary = result[0]['summary_text']
        else:
            return "지원되지 않는 언어입니다."
    except Exception as e:
        return f"요약 중 오류가 발생했습니다: {e}"

    return summary

# 키워드 추출을 위한 KeyBERT 모델 로드
model_english = SentenceTransformer('all-MiniLM-L6-v2')
kw_model_english = KeyBERT(model=model_english)
model_korean = SentenceTransformer('xlm-r-bert-base-nli-stsb-mean-tokens')
kw_model_korean = KeyBERT(model=model_korean)

def extract_keywords(text, language):
    """
    입력 텍스트에서 키워드를 추출하는 함수
    """
    if len(text.split()) < 20:
        return ["입력 텍스트가 너무 짧습니다. 더 긴 텍스트를 입력해주세요."]
    
    try:
        if language == 'English':
            keywords = kw_model_english.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=5)
        elif language == 'Korean':
            keywords = kw_model_korean.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=5)
        else:
            return ["지원되지 않는 언어입니다."]
    except Exception as e:
        return [f"키워드 추출 중 오류가 발생했습니다: {e}"]

    # 키워드 리스트에 번호 매기기
    return [f"{i + 1}. {keyword[0]}" for i, keyword in enumerate(keywords)]

# Gradio 인터페이스 생성
def create_gradio_interface():
    """
    Gradio 인터페이스를 생성하고 실행하는 함수
    """
    # 텍스트 요약 인터페이스
    summarize_interface = gr.Interface(
        fn=summarize_text, 
        inputs=[gr.Textbox(lines=10, label="텍스트 입력"), gr.Dropdown(['English'], label="언어 선택")], 
        outputs="text", 
        title="텍스트 요약"
    )
    
    # 키워드 추출 인터페이스
    keywords_interface = gr.Interface(
        fn=extract_keywords, 
        inputs=[gr.Textbox(lines=10, label="텍스트 입력"), gr.Dropdown(['English', 'Korean'], label="언어 선택")], 
        outputs="json", 
        title="키워드 추출"
    )
    
    # 여러 인터페이스를 하나의 앱으로 결합
    demo = gr.TabbedInterface(
        interface_list=[summarize_interface, keywords_interface],
        tab_names=["텍스트 요약", "키워드 추출"]
    )
    
    # 인터페이스 실행
    demo.launch()

# Gradio 인터페이스 실행
create_gradio_interface()
