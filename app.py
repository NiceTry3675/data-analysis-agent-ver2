import io
import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

def generate_target_schema(api_key: str, all_previews: List[str]) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """
    여러 파일의 프리뷰를 보고 공통 목표 스키마(Target Schema)를 제안.
    반환: (schema_dict, raw_response_text, error_message)
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
    raw_text = ""
    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
        )
        raw_text = resp.text
        schema = json.loads(extract_json_block(raw_text))
        if not isinstance(schema, dict):
            raise ValueError("LLM 응답이 dict 형식이 아닙니다.")
        return schema, raw_text, None
    except Exception as e:
        return {"columns": []}, raw_text, str(e)


def generate_transform_code(
    api_key: str,
    file_name: str,
    sheet_name: str,
    df_preview: str,
    target_columns: List[str]
) -> Tuple[str, str, Optional[str]]:
    """
    Raw Data -> Target Schema로 변환하는 파이썬 코드 작성.
    반환: (code_str, raw_response_text, error_message)
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
   - 헤더나 데이터 시작 행을 찾지 못해도 `ValueError`를 던지지 말고, 합리적인 기본 인덱스를 사용하거나 빈 DataFrame이라도 반환하세요.

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

    raw_text = ""
    try:
        resp = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=system_instruction + "\n" + prompt,
        )
        raw_text = getattr(resp, "text", "") or ""
        if not raw_text.strip():
            raise ValueError("LLM 응답이 비어 있습니다. API 키/쿼터 또는 네트워크를 확인하세요.")

        code = extract_python_code(raw_text)
        if not code.strip():
            raise ValueError("LLM이 코드를 반환하지 않았습니다. 프롬프트를 다시 시도하거나 프리뷰 행 수를 줄여보세요.")
        return code, raw_text, None
    except Exception as e:
        return "", raw_text, str(e)


# ==============================================================================
# 3. 코드 실행기
# ==============================================================================


def execute_user_code(code_str: str, df_raw: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict], str]:
    """사용자/LLM 코드 실행 래퍼. 에러를 문자열로 반환한다."""
    local_scope = {"pd": pd, "df_raw": df_raw, "np": np, "re": re}

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
    
    # API 키 안내 메시지
    st.sidebar.info("💡 Key 입력하지 않아도 사용 가능합니다.")
    
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    uploaded_files = st.sidebar.file_uploader(
        "엑셀 파일 업로드 (여러 개)", type=["xls", "xlsx"], accept_multiple_files=True
    )
    st.sidebar.caption("⚠️ **주의**: '같은 종류의 데이터지만 양식이 다른 파일'만 업로드해주세요. 성격이 아예 다른 데이터(예: 인구수 vs 매출)는 제외해야 정확도가 높습니다. 여러 시트가 있는 파일인 경우, 통일하고 싶지 않은 시트가 포함된 파일은 제안/변환 단계에서 제외 할 수 있습니다.")

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
    if "llm_logs" not in st.session_state:
        st.session_state["llm_logs"] = []

    # 1. 파일 로딩
    all_data = {}
    if uploaded_files:
        for file_idx, uf in enumerate(uploaded_files):
            try:
                dfs = read_excel_sheets(uf)
                for sname, df in dfs.items():
                    # (업로드 순서 index, 파일명, 시트명)으로 키를 만들어 동일 파일명 충돌 방지
                    all_data[(file_idx, uf.name, sname)] = df
            except Exception as e:
                st.sidebar.error(f"{uf.name} 로드 실패: {e}")
                with st.sidebar.expander("상세 에러"):
                    st.code(traceback.format_exc())

    if not all_data:
        st.info("파일을 업로드해주세요.")
        return

    # 2. 타겟 스키마 정의
    st.header("1️⃣ 공통 타겟 스키마 (Target Schema) 정의")
    
    # --- 스키마 선택 영역 (전체 폭)
    schema_keys = list(all_data.keys())
    label_map = {k: f"{k[0]+1}) {k[1]}::{k[2]}" for k in schema_keys}
    default_selection = schema_keys[: min(3, len(schema_keys))]
    selected_for_schema = st.multiselect(
        "스키마 제안에 사용할 시트 선택 (선택 없으면 전체 사용)",
        options=schema_keys,
        default=default_selection,
        format_func=lambda k: label_map.get(k, str(k)),
        help="긴 파일/시트 이름을 전부 표시하기 위해 영역을 넓혔습니다. 필요 시 원하는 시트만 선택하세요.",
    )

    col1, col2 = st.columns([1, 2.2])
    with col1:
        if st.button("🤖 AI 스키마 자동 제안", use_container_width=True):
            with st.spinner("데이터 샘플 분석 중..."):
                target_keys = selected_for_schema or schema_keys
                samples = []
                for k in target_keys:
                    df = all_data[k]
                    samples.append(get_dataframe_preview_markdown(df, rows=10))
                
                schema_def, raw_resp, err = generate_target_schema(api_key, samples)
                st.session_state["llm_logs"].append({"type": "schema", "raw": raw_resp, "error": err})

                if err:
                    st.error("스키마 생성 실패")
                    st.error(err)
                    with st.expander("AI 원본 응답"):
                        st.text(raw_resp or "(빈 응답)")
                else:
                    st.session_state["target_schema"] = schema_def
                    st.success("스키마 제안 완료 (편집 가능)")

    with col2:
        current_schema = st.session_state.get("target_schema")
        if current_schema is None:
            current_schema = {"columns": []}
            
        schema_text = st.text_area(
            "스키마 정의 (JSON 편집 가능)", 
            value=json.dumps(current_schema, indent=2, ensure_ascii=False),
            height=240,
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
        st.warning("타겟 스키마에 columns가 없습니다. JSON에 columns 리스트를 추가해주세요.")
        return

    st.success(f"목표 컬럼: {target_columns}")

    # 3. 개별 파일 변환 코드 생성 및 실행
    st.header("2️⃣ 파일별 변환 및 병합")

    entries = list(all_data.items())
    options = [key for key, _ in entries]

    # 전체 자동 실행 (선택한 시트만)
    st.markdown("#### ⚡ 선택 시트 일괄 변환")
    auto_run_targets = st.multiselect(
        "일괄 실행 대상 시트 선택",
        options=options,
        default=options,
        format_func=lambda k: f"{k[0]+1}) {k[1]}::{k[2]}",
        help="여기서 선택한 시트만 코드 생성+실행을 순차 처리합니다."
    )
    auto_run = st.button("선택 시트 코드 생성 + 실행", help="선택한 시트를 현재 스키마로 순차 실행합니다.")

    valid_dfs = []

    if auto_run:
        run_list = [item for item in entries if item[0] in auto_run_targets]
        if not run_list:
            st.warning("일괄 실행할 시트를 선택해주세요.")
        else:
            success, fail = [], []
            progress = st.progress(0)
            total = len(run_list)
            for idx, (key, df_raw) in enumerate(run_list, start=1):
                file_idx, fname, sname = key
                unique_id = f"{file_idx}:{fname}::{sname}"
                try:
                    preview = get_dataframe_preview_markdown(df_raw)
                    code, raw_resp, err = generate_transform_code(api_key, fname, sname, preview, target_columns)
                    st.session_state["llm_logs"].append({"type": "code", "file": fname, "sheet": sname, "raw": raw_resp, "error": err})
                    if err:
                        fail.append((fname, sname, f"생성 실패: {err}"))
                        continue

                    st.session_state["generated_codes"][unique_id] = code
                    st.session_state[f"edit_{unique_id}"] = code

                    df_res, meta, exec_err = execute_user_code(code, df_raw)
                    if exec_err:
                        fail.append((fname, sname, f"실행 오류: {exec_err.splitlines()[-1] if exec_err else exec_err}"))
                        continue

                    missing = [c for c in target_columns if c not in df_res.columns]
                    if missing:
                        fail.append((fname, sname, f"목표 컬럼 누락: {missing}"))
                        st.session_state["results"].pop(unique_id, None)
                        continue

                    df_res = df_res[target_columns]
                    df_res["_source_file"] = fname
                    df_res["_source_sheet"] = sname
                    st.session_state["results"][unique_id] = df_res
                    success.append((fname, sname, len(df_res)))
                finally:
                    progress.progress(idx / total)

            if success:
                st.success(f"자동 변환 성공 {len(success)}건")
                for f, s, rows in success:
                    st.write(f"✅ {f} / {s} ({rows}행)")
            if fail:
                st.error(f"실패 {len(fail)}건")
                for f, s, msg in fail:
                    st.write(f"❌ {f} / {s}: {msg}")

    # 수동 선택 영역
    selected_key = st.selectbox(
        "수동으로 변환할 파일/시트 선택",
        options=options,
        format_func=lambda k: f"{k[0]+1}) {k[1]}::{k[2]}"
    )

    for key, df_raw in entries:
        file_idx, fname, sname = key
        unique_id = f"{file_idx}:{fname}::{sname}"

        if key != selected_key:
            if unique_id in st.session_state["results"]:
                valid_dfs.append(st.session_state["results"][unique_id])
            continue

        st.subheader(f"선택된 시트: {fname} / {sname}")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("#### Raw Data Preview")
            safe_dataframe_display(df_raw.head(15))

            if st.button(f"코드 생성 ({sname})", key=f"gen_{unique_id}"):
                with st.spinner("변환 코드 작성 중..."):
                    preview = get_dataframe_preview_markdown(df_raw)
                    code, raw_resp, err = generate_transform_code(api_key, fname, sname, preview, target_columns)
                    st.session_state["llm_logs"].append({"type": "code", "file": fname, "sheet": sname, "raw": raw_resp, "error": err})
                    if err:
                        st.error(f"코드 생성 실패: {err}")
                        with st.expander("AI 원본 응답"):
                            st.text(raw_resp or "(빈 응답)")
                    else:
                        st.session_state["generated_codes"][unique_id] = code
                        st.session_state[f"edit_{unique_id}"] = code
                        st.success("코드 생성 완료. 필요하면 수정 후 실행하세요.")

        with c2:
            st.markdown("#### Transformation Code")
            code_key = f"edit_{unique_id}"
            generated_code = st.session_state["generated_codes"].get(unique_id, "")
            if code_key not in st.session_state and generated_code:
                st.session_state[code_key] = generated_code
            edited_code = st.text_area("Python Code", st.session_state.get(code_key, generated_code), height=300, key=code_key)
            if not edited_code.strip():
                st.info("코드가 비어 있습니다. 좌측에서 '코드 생성'을 다시 눌러주세요. 에러가 있다면 아래 LLM 응답 로그를 확인하세요.")
            st.session_state["generated_codes"][unique_id] = edited_code

            if st.button(f"실행 ({sname})", key=f"exec_{unique_id}"):
                df_res, meta, err = execute_user_code(edited_code, df_raw)
                if err:
                    st.error("코드 실행 오류")
                    st.code(err, language="text")
                else:
                    missing = [c for c in target_columns if c not in df_res.columns]
                    if missing:
                        st.warning(f"⚠️ 목표 컬럼 누락 -> {missing}")
                        st.session_state["results"].pop(unique_id, None)
                    else:
                        df_res = df_res[target_columns]
                        df_res["_source_file"] = fname
                        df_res["_source_sheet"] = sname
                        st.session_state["results"][unique_id] = df_res
                        st.success("변환 성공! 아래에서 병합 가능")
                        safe_dataframe_display(df_res.head(5))

        if unique_id in st.session_state["results"]:
            valid_dfs.append(st.session_state["results"][unique_id])

    # 진행 현황 요약
    st.divider()
    st.markdown("### 진행 현황")
    status_rows = []
    for (idx, fname, sname) in all_data.keys():
        uid = f"{idx}:{fname}::{sname}"
        status_rows.append({
            "#": idx + 1,
            "파일": fname,
            "시트": sname,
            "코드 생성": "✅" if uid in st.session_state["generated_codes"] and st.session_state["generated_codes"][uid].strip() else "⬜",
            "실행 완료": "✅" if uid in st.session_state["results"] else "⬜",
        })
    st.dataframe(pd.DataFrame(status_rows))

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

    # 디버그: LLM 원본 응답 모아보기
    with st.expander("LLM 응답 로그"):
        if not st.session_state["llm_logs"]:
            st.write("로그 없음")
        else:
            for i, log in enumerate(reversed(st.session_state["llm_logs"])):
                label = f"{i+1}. {log.get('type', '')} | {log.get('file', '')} {log.get('sheet', '')}"
                st.markdown(f"**{label}**")
                if log.get("error"):
                    st.error(log["error"])
                st.code(log.get("raw", "(raw 없음)"))


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
