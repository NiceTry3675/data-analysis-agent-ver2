import io
import json
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from google import genai

# ==============================================================================
# 1. 유틸리티 & 헬퍼 함수
# ==============================================================================


def read_excel_sheets(uploaded_file) -> Dict[str, pd.DataFrame]:
    """엑셀 파일을 읽어 {sheet_name: df} 형태로 반환 (헤더 없이 raw read)"""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    
    ext = uploaded_file.name.lower().split(".")[-1]
    engine = "openpyxl"
    if ext == "xls":
        engine = "xlrd"

    # header=None으로 읽어서 위치 기반 처리를 가능하게 함
    dfs = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=None,
        header=None,
        engine=engine
    )
    return dfs


def get_dataframe_preview_markdown(df: pd.DataFrame, rows: int = 20) -> str:
    """LLM에게 보여줄 DataFrame의 Markdown 표현"""
    preview_df = df.head(rows).copy()
    # Arrow 오류 방지를 위해 문자열로 변환
    return preview_df.fillna("").astype(str).to_markdown(index=True)


def extract_python_code(text: str) -> str:
    """마크다운 코드 블록에서 파이썬 코드만 추출"""
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()
    return text.strip()


def extract_json_block(text: str) -> str:
    """마크다운 코드 블록 혹은 텍스트에서 JSON 추출"""
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()
    
    # JSON 블록이 명시적이지 않은 경우 { } 찾기
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return text


def safe_dataframe_display(df: pd.DataFrame, height: int = None):
    """
    st.dataframe을 안전하게 렌더링하는 래퍼.
    Arrow Serialization 에러 발생 시 문자열로 변환하여 재시도.
    """
    kwargs = {}
    if height is not None:
        kwargs["height"] = height

    try:
        st.dataframe(df, **kwargs)
    except Exception as e:
        st.warning(f"⚠️ 기본 데이터프레임 렌더링 실패 (Arrow 호환성 문제). 텍스트 모드로 표시합니다. 에러: {e}")
        try:
            st.dataframe(df.astype(str), **kwargs)
        except Exception as e2:
            st.error(f"❌ 데이터 표시 실패: {e2}")
            st.code(str(df.head(20)))  # 최후의 수단: repr 문자열 출력


# ==============================================================================
# 2. LLM 로직 (Schema 설계 + Code Gen)
# ==============================================================================

def generate_target_schema(api_key: str, all_previews: List[str]) -> Dict[str, Any]:
    """
    여러 파일의 프리뷰를 보고 공통 목표 스키마(Target Schema)를 제안
    """
    client = genai.Client(api_key=api_key)
    
    previews_text = "\n\n".join([f"--- File Sample {i+1} ---\n{p}" for i, p in enumerate(all_previews[:3])])

    prompt = f"""
당신은 데이터 아키텍트입니다. 여러 개의 비정형 엑셀 파일들을 분석하여 하나로 통합하기 위한 '공통 타겟 스키마(Target Schema)'를 설계해야 합니다.

목표:
1. 모든 파일에서 공통적으로 추출 가능한 핵심 분석 항목(컬럼)을 정의하세요.
2. 컬럼명은 영문 스네이크 케이스(snake_case)로 통일하세요.
3. 각 컬럼의 데이터 타입과 설명을 포함하세요.

입력 데이터 샘플:
{previews_text}

응답 형식 (JSON):
```json
{{
    "table_name": "integrated_data",
    "columns": [
        {{"name": "region", "type": "string", "description": "지역명"}},
        {{"name": "date", "type": "date", "description": "기준 일자"}},
        {{"name": "population", "type": "int", "description": "인구 수"}}
    ]
}}
```
반드시 위 JSON 형식만 출력하세요.
"""
    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
    )
    
    text = resp.text
    try:
        return json.loads(extract_json_block(text))
    except json.JSONDecodeError as e:
        # 파싱 실패 시 에러 정보를 담아 리턴 (UI에서 처리)
        return {"columns": [], "_error": f"JSON 파싱 실패: {e}", "_raw_response": text}
    except Exception as e:
        return {"columns": [], "_error": f"알 수 없는 오류: {e}", "_raw_response": text}


def generate_transform_code(
    api_key: str, 
    file_name: str, 
    sheet_name: str, 
    df_preview: str,
    target_columns: List[str]
) -> str:
    """
    Raw Data -> Target Schema로 변환하는 파이썬 코드 작성
    """
    client = genai.Client(api_key=api_key)
    
    target_cols_str = ", ".join([f"'{c}'" for c in target_columns])
    
    system_instruction = f"""
당신은 Python Pandas 전문가입니다.
Raw Excel Data를 정제하여, 반드시 **[Target Schema]**에 정의된 컬럼을 가진 DataFrame으로 변환하는 `transform(df)` 함수를 작성하세요.

### 필수 요구사항
1. **함수 시그니처**: `def transform(df: pd.DataFrame) -> (pd.DataFrame, dict):`
2. **Target Schema 준수**: 반환되는 `df_clean`은 다음 컬럼들을 반드시 포함해야 합니다: [{target_cols_str}]
   - 데이터에 해당 정보가 없다면 `None`이나 기본값으로 채우세요.
   - 불필요한 컬럼은 과감히 버리세요.
3. **메타데이터**: 제목, 단위 등은 별도 dict로 반환.
4. **전처리**: 
   - 헤더 탐색, 불필요한 상단 행 제거
   - '합계', '소계' 등 통계 행 제거
   - Wide to Long (Melt) 변환 적극 활용

### 출력 형식
마크다운 코드 블럭(```python ... ```) 안에 파이썬 코드를 작성하세요.

### 코드 템플릿
```python
import pandas as pd
import numpy as np

def transform(df):
    metadata = {{}}
    
    # 1. 메타데이터 추출
    # ...
    
    # 2. 헤더 찾기 및 데이터 슬라이싱
    # ...
    
    # 3. 컬럼 매핑 및 데이터 정제
    # ...
    
    # 4. Target Schema 맞추기 (필수 단계)
    # 필요한 컬럼 생성 및 선택
    # df_clean = ...
    
    return df_clean, metadata
```
"""

    prompt = f"""
### 처리할 파일
* 파일: {file_name} / 시트: {sheet_name}

### 데이터 미리보기
```markdown
{df_preview}
```

위 데이터를 [{target_cols_str}] 컬럼을 가진 데이터프레임으로 변환하는 코드를 작성해주세요.
"""

    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=system_instruction + "\n" + prompt,
    )
    
    return extract_python_code(resp.text)


# ==============================================================================
# 3. 코드 실행기
# ==============================================================================


def execute_user_code(code_str: str, df_raw: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict], str]:
    local_scope = {"pd": pd, "df_raw": df_raw, "np": pd.np} 
    
    try:
        exec(code_str, globals(), local_scope)
        if "transform" not in local_scope:
            return None, None, "Error: 'transform' 함수가 정의되지 않았습니다."
            
        transform_func = local_scope["transform"]
        df_clean, metadata = transform_func(df_raw.copy())
        
        if not isinstance(df_clean, pd.DataFrame):
            return None, None, "Error: 반환값은 DataFrame이어야 합니다."
            
        return df_clean, metadata, ""
        
    except Exception:
        return None, None, traceback.format_exc()


# ==============================================================================
# 4. Streamlit UI
# ==============================================================================


def main_app():
    st.set_page_config(page_title="AI 통합 데이터 정제기", layout="wide")
    st.title("🧩 AI 통합 데이터 정제기 (Many to One)")
    st.markdown("""
    여러 개의 **비정형 엑셀 파일**을 AI가 작성한 코드를 통해 **하나의 통일된 CSV**로 합칩니다.
    1. AI가 공통 스키마(Target Schema)를 제안합니다.
    2. 각 파일별로 스키마에 맞추는 변환 코드를 생성합니다.
    3. 결과를 하나로 병합하여 다운로드합니다.
    """)

    # --- 설정 ---
    st.sidebar.header("설정")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    uploaded_files = st.sidebar.file_uploader(
        "엑셀 파일 업로드 (여러 개)", type=["xls", "xlsx"], accept_multiple_files=True
    )

    if not api_key:
        st.warning("Gemini API Key가 필요합니다.")
        return
        
    # Session State 초기화
    if "target_schema" not in st.session_state:
        st.session_state["target_schema"] = {"columns": []}
    if "generated_codes" not in st.session_state:
        st.session_state["generated_codes"] = {}
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    # 1. 파일 로딩
    all_data = {}
    if uploaded_files:
        for uf in uploaded_files:
            try:
                dfs = read_excel_sheets(uf)
                for sname, df in dfs.items():
                    all_data[(uf.name, sname)] = df
            except Exception as e:
                st.sidebar.error(f"{uf.name} 로드 실패: {e}")
                with st.sidebar.expander("상세 에러"):
                    st.code(traceback.format_exc())

    if not all_data:
        st.info("파일을 업로드해주세요.")
        return

    # 2. 타겟 스키마 정의
    st.header("1️⃣ 공통 타겟 스키마 (Target Schema) 정의")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🤖 AI 스키마 자동 제안"):
            with st.spinner("데이터 샘플 분석 중..."):
                samples = []
                for k, df in list(all_data.items())[:3]:
                    samples.append(get_dataframe_preview_markdown(df, rows=10))
                
                schema_def = generate_target_schema(api_key, samples)
                
                # 에러 체크
                if "_error" in schema_def:
                    st.error("스키마 생성 실패")
                    st.error(schema_def["_error"])
                    with st.expander("AI 원본 응답"):
                        st.text(schema_def.get("_raw_response", ""))
                else:
                    st.session_state["target_schema"] = schema_def

    with col2:
        current_schema = st.session_state.get("target_schema")
        if current_schema is None:
            current_schema = {"columns": []}
            
        schema_text = st.text_area(
            "스키마 정의 (JSON 편집 가능)", 
            value=json.dumps(current_schema, indent=2, ensure_ascii=False),
            height=200
        )
        try:
            parsed = json.loads(schema_text)
            if isinstance(parsed, dict):
                st.session_state["target_schema"] = parsed
        except Exception as e:
            st.error(f"JSON 형식 오류: {e}")

    schema = st.session_state.get("target_schema")
    if not isinstance(schema, dict):
        schema = {"columns": []}
    
    target_columns = [c["name"] for c in schema.get("columns", []) if isinstance(c, dict) and "name" in c]
    
    if not target_columns:
        st.warning("위에서 타겟 스키마를 정의하거나 AI 제안을 받아주세요.")
        return

    st.success(f"목표 컬럼: {target_columns}")

    # 3. 개별 파일 변환 코드 생성 및 실행
    st.header("2️⃣ 파일별 변환 및 병합")
    
    tabs = st.tabs([f"{f}::{s}" for f, s in all_data.keys()])
    
    valid_dfs = []

    for i, (key, df_raw) in enumerate(all_data.items()):
        fname, sname = key
        unique_id = f"{fname}::{sname}"
        
        with tabs[i]:
            c1, c2 = st.columns([1, 1])
            
            # 코드 생성
            with c1:
                st.markdown("#### Raw Data Preview")
                # 안전한 렌더링 사용 (Raw 데이터는 보통 타입이 섞여 있으므로 주의 필요)
                safe_dataframe_display(df_raw.head(15))
                
                if st.button(f"코드 생성 ({sname})", key=f"gen_{unique_id}"):
                    with st.spinner("변환 코드 작성 중..."):
                        preview = get_dataframe_preview_markdown(df_raw)
                        code = generate_transform_code(api_key, fname, sname, preview, target_columns)
                        st.session_state["generated_codes"][unique_id] = code
                        st.rerun()
            
            # 코드 실행
            with c2:
                st.markdown("#### Transformation Code")
                code_val = st.session_state["generated_codes"].get(unique_id, "")
                edited_code = st.text_area("Python Code", code_val, height=300, key=f"edit_{unique_id}")
                st.session_state["generated_codes"][unique_id] = edited_code
                
                if st.button(f"실행 ({sname})", key=f"exec_{unique_id}"):
                    df_res, meta, err = execute_user_code(edited_code, df_raw)
                    if err:
                        st.error("코드 실행 오류")
                        st.code(err, language="text")
                    else:
                        # 스키마 검증
                        missing = [c for c in target_columns if c not in df_res.columns]
                        if missing:
                            st.warning(f"⚠️ 주의: 목표 컬럼 누락 -> {missing}")
                        else:
                            # 컬럼 순서 정렬
                            df_res = df_res[target_columns] 
                            
                            df_res["_source_file"] = fname
                            df_res["_source_sheet"] = sname
                            
                            st.session_state["results"][unique_id] = df_res
                            st.success("변환 성공!")
                            safe_dataframe_display(df_res.head(5))
                            st.rerun()

        if unique_id in st.session_state["results"]:
            valid_dfs.append(st.session_state["results"][unique_id])

    # 4. 최종 병합 및 다운로드
    st.divider()
    st.header("3️⃣ 최종 병합 및 다운로드")
    
    if valid_dfs:
        try:
            final_df = pd.concat(valid_dfs, ignore_index=True)
            st.markdown(f"### 📦 총 {len(valid_dfs)}개 파일 병합 완료 ({len(final_df)} 행)")
            safe_dataframe_display(final_df.head(20))
            
            csv_bytes = final_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 통합 CSV 다운로드",
                data=csv_bytes,
                file_name="merged_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error("병합 중 오류 발생")
            st.code(traceback.format_exc())
    else:
        st.info("아직 변환된 데이터가 없습니다. 각 탭에서 코드를 생성하고 실행해주세요.")


def main():
    try:
        main_app()
    except Exception as e:
        st.error("🚨 애플리케이션 실행 중 예기치 못한 오류가 발생했습니다.")
        st.error(str(e))
        with st.expander("상세 오류 로그 (Traceback)"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
