import base64
import os
from datetime import datetime
from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 환경 설정
# ==========================================
UBUNTU_IP = "192.168.0.201" 
SERVER_URL = f"http://{UBUNTU_IP}:11435"

VISION_MODEL = "qwen2.5vl:32b" 
REFINE_MODEL = "qwen3:32b"

class AgentState(TypedDict):
    image_path: str
    raw_markdown: Optional[str]
    final_result: Optional[str]
    error: Optional[str]

# ==========================================
# 3. 노드 구현 (프롬프트 전략 수정)
# ==========================================

def vision_analysis_node(state: AgentState):
    print(f"\n🎨 [Node 1] 정밀 고해상도 스캔 중... (Model: {VISION_MODEL})")
    try:
        with open(state['image_path'], "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        llm = ChatOllama(model=VISION_MODEL, base_url=SERVER_URL, temperature=0)
        
        # 🎯 전략: 좌우 분리 요청 및 키워드 기반 강박적 추출
        instruction = """너는 세계 최고의 OCR 전문가야. 이미지를 아주 미세한 단위로 분석해서 마크다운으로 변환해.

        [중요: 레이아웃 처리]
        만약 이미지에 영수증이 좌/우 두 개가 붙어 있다면, 반드시 '영수증 1(좌)', '영수증 2(우)'로 나누어 각각 처음부터 끝까지 추출해.

        [중요: 필수 추출 체크리스트 - 하나라도 누락하면 안 됨]
        1. 가맹점 정보: 상호명, 사업자번호, 주소, 전화번호 (블러 처리된 부분은 '판독불가'로 표시)
        2. 품목 정보: 메뉴명, 단가, 수량, 금액 (표 형식 유지)
        3. 세금 정보: 부가세 과세 물품가액, 부가세, 면세 물품가액
        4. 결제 정보 (가장 중요): 
           - 카드번호: (예: 5417-****-****-****)
           - 카드명: (예: NH농협카드, 국민카드 등)
           - 가맹점번호: (영수증 하단에 기재된 번호)
           - 매입사명: (예: 비씨카드사 등)
           - 승인번호: (반드시 8자리 숫자를 정확히 찾아낼 것)
           - 할부개월: (일시불인지 개월수가 있는지 정확히 기재)

        [금지 사항]
        - 절대로 카드번호 자리에 결제 금액을 넣지 마.
        - '이름1', '상품1' 같은 가짜 데이터를 생성하지 마. 보이는 그대로만 적어.
        - 숫자는 콤마(,)까지 그대로 보존해."""

        content_list = [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]
        
        message = HumanMessage(content=content_list)
        response = llm.invoke([message])
        return {"raw_markdown": response.content}
    except Exception as e:
        print(f"❌ Vision 노드 에러: {e}")
        return {"error": f"Vision Analysis Failed: {str(e)}"}


def logic_refinement_node(state: AgentState):
    if state.get('error'): return state
    print(f"🧠 [Node 2] 데이터 정합성 검증 및 교정... (Model: {REFINE_MODEL})")
    
    llm = ChatOllama(model=REFINE_MODEL, base_url=SERVER_URL, temperature=0)
    
    # 🎯 전략: 추출된 데이터에서 승인번호, 카드번호 등의 위치가 올바른지 다시 검토
    prompt = f"""
    너는 영수증 데이터 검증 AI야. Vision 모델이 추출한 데이터에서 오류를 찾아 수정하고 마크다운을 완성해.

    [검토 포인트]
    1. '카드번호' 항목에 '480,000' 같은 금액이 들어가 있지 않은지 확인해. 금액이라면 '결제금액'으로 옮기고 카드번호는 이미지에서 다시 찾아보거나 '미표기'로 처리해.
    2. '승인번호'가 숫자 8자리 혹은 그 근처인지 확인하고, 엉뚱한 텍스트가 섞였다면 숫자만 남겨.
    3. 좌측 영수증과 우측 영수증의 데이터가 서로 섞이지 않았는지 철저히 분리해.
    4. 사업자번호, 상호명 등 상단 정보가 '판독불가'인 경우, 영수증 하단이나 다른 곳에 정보가 있는지 재확인해.

    --- 추출된 원본 데이터 ---
    {state['raw_markdown']}
    """
    
    response = llm.invoke(prompt)
    return {"final_result": response.content}

# ==========================================
# 4. 그래프 구성 및 실행부 (동일)
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("vision_analysis", vision_analysis_node)
workflow.add_node("logic_refinement", logic_refinement_node)
workflow.set_entry_point("vision_analysis")
workflow.add_edge("vision_analysis", "logic_refinement")
workflow.add_edge("logic_refinement", END)
app = workflow.compile()

def run_analysis(img_path: str):
    if not os.path.exists(img_path):
        print(f"❌ 파일을 찾을 수 없습니다: {img_path}")
        return

    start_time = datetime.now()
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    initial_input = {"image_path": img_path}
    print(f"🚀 랭그래프 에이전트 가동... (시작: {start_str})")
    
    final_state = app.invoke(initial_input)
    
    end_time = datetime.now()
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    if final_state.get('error'):
        print(f"⚠️ 에러 발생: {final_state['error']}")
    else:
        result_text = final_state['final_result']
        output_filename = f"{img_path}.txt"
        
        file_content = f"작업 시작: {start_str}\n"
        file_content += "="*60 + "\n"
        file_content += f"입력 이미지: {os.path.abspath(img_path)}\n"
        file_content += "="*60 + "\n"
        file_content += result_text + "\n"
        file_content += "="*60 + "\n"
        file_content += f"작업 종료: {end_str}\n"
        file_content += f"총 소요시간: {end_time - start_time}\n"

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(file_content)
        
        print(f"\n✅ 분석 완료! 파일이 생성되었습니다: {output_filename}")
        print(result_text)

if __name__ == "__main__":
    target_image = "./img/test1.png" # 파일 확장자 확인 (png/jpg)
    run_analysis(target_image)