import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from google import genai


# ======================================
# 1. 엑셀 읽기 & 시트 프로파일 생성
# ======================================


def read_all_sheets(uploaded_file) -> Dict[str, pd.DataFrame]:
    """
    Streamlit UploadedFile → {sheet_name: DataFrame(header=None)} 로 읽기.
    """
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    ext = uploaded_file.name.lower().split(".")[-1]
    engine = "openpyxl"
    if ext == "xls":
        engine = "xlrd"  # 구버전 xls 용

    dfs = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=None,
        header=None,
        engine=engine,
    )
    return dfs


def profile_sheet_for_llm(
    df_raw: pd.DataFrame,
    file_name: str,
    sheet_name: str,
    max_rows: int = 12,
    max_cols: int = 12,
    sample_per_col: int = 5,
) -> Dict[str, Any]:
    """
    LLM에 넘길 시트 프로파일:
    - 상위 N행 값 (텍스트)
    - 컬럼별 타입/샘플 요약
    """
    n_rows, n_cols = df_raw.shape
    preview_rows: List[List[str]] = []
    for i in range(min(max_rows, n_rows)):
        row: List[str] = []
        for j in range(min(max_cols, n_cols)):
            val = df_raw.iat[i, j]
            if pd.isna(val):
                row.append("")
            else:
                row.append(str(val))
        preview_rows.append(row)

    columns_profile: List[Dict[str, Any]] = []
    for col_idx in range(min(max_cols, n_cols)):
        col = df_raw.iloc[:, col_idx]
        non_null = col.dropna()
        head_samples = non_null.head(sample_per_col).astype(str).tolist()
        unique_ratio = float(non_null.nunique() / non_null.size) if non_null.size > 0 else 0.0

        if pd.api.types.is_numeric_dtype(col):
            logical_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(col):
            logical_type = "datetime"
        else:
            logical_type = "string"

        columns_profile.append(
            {
                "index": col_idx,
                "pandas_dtype": str(col.dtype),
                "logical_type_guess": logical_type,
                "non_null_ratio": float(non_null.size / len(col)) if len(col) > 0 else 0.0,
                "unique_ratio": unique_ratio,
                "sample_values": head_samples,
            }
        )

    return {
        "file_name": file_name,
        "sheet_name": sheet_name,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "preview_rows": preview_rows,
        "columns": columns_profile,
    }


# ======================================
# 2. LLM 프롬프트 & 호출
# ======================================


def build_llm_prompt(sheet_profiles: List[Dict[str, Any]]) -> str:
    """
    LLM에게:
      - target_tables (통일 스키마)
      - sheet_mappings (시트별 전처리/매핑/메타데이터 추출)
    을 설계하게 하는 프롬프트.
    """

    profiles_json = json.dumps(sheet_profiles, ensure_ascii=False, indent=2)

    dsl_spec = r"""
당신은 엑셀 리포트를 정규화하여 깔끔한 CSV + 메타데이터(JSON)로 만드는 데이터 엔지니어입니다.

목표:
- CSV에는 분석에 필요한 **핵심 필드만** 담습니다.
- CSV로 표현하기 지저분한 정보(멀티 헤더, 제목, 기준일, 단위, 주석 등)는 메타데이터(JSON)로 뺍니다.
- 여러 다른 형식의 시트라도, 가능한 한 **공통 테이블 스키마**로 통일합니다.

입력:
- 여러 개의 엑셀 파일/시트에 대해, 상위 일부 행과 컬럼 요약을 담은 프로파일이 주어집니다.

출력:
- 1) target_tables: 통일된 테이블 스키마 정의
- 2) sheet_mappings: 각 (file_name, sheet_name)을 어떤 테이블로, 어떻게 변환할지 규칙 정의

### 1) target_tables 구조

```json
"target_tables": [
  {
    "name": "string (테이블 이름, 예: population_main)",
    "description": "이 테이블이 무엇을 표현하는지 한국어 설명",
    "columns": [
      {
        "name": "string (영문 스네이크케이스 권장, 예: region_name)",
        "dtype": "string | int | float | bool | date",
        "role": "data | metadata | both",
        "description": "컬럼 의미 (한국어)"
      }
    ]
  }
]
```

* role이 의미하는 것:

  * "data": CSV에 포함 (분석용 핵심 데이터)
  * "metadata": metadata.json에만 포함 (행마다 반복하기 애매한 정보)
  * "both": CSV에도 컬럼으로 두고, metadata.json에도 요약/설명에 포함

### 2) sheet_mappings 구조

```json
"sheet_mappings": [
  {
    "file_name": "원본 엑셀 파일명 (프로파일에 나온 값과 동일하게)",
    "sheet_name": "원본 시트명 (프로파일에 나온 값과 동일하게)",
    "table_name": "target_tables 중 하나의 name",

    "preprocess": {
      "header_row": 5,               // 0-based, 이 행을 컬럼 헤더로 사용
      "drop_top_rows": 0,            // header_row 바로 아래에서 추가로 버릴 행 수
      "drop_bottom_rows": 0,         // 마지막에서 몇 행을 버릴지
      "drop_empty_rows": true,       // 전부 비어있는 행 drop 여부
      "drop_empty_columns": true,    // 전부 비어있는 열 drop 여부
      "drop_rows_matching": {
        "column_index": 0,           // 첫 번째 열 기준
        "equals": ["합계", "전월합계"] // 이 값(또는 텍스트)인 행은 데이터에서 제외
      }
    },

    "melt": {
      "enabled": true,
      "id_columns": ["연령", "성별"],        // header 적용 후 기준이 되는 id 컬럼들
      "value_columns": "all_except_id",      // 또는 ["진주시", "문산읍", ...]
      "variable_name": "지역",               // melt 후 지역 이름이 들어갈 컬럼명
      "value_name": "인구"                   // melt 후 값이 들어갈 컬럼명
    },

    "column_mapping": {
      "연령": "age_label",      // 현재 컬럼명(또는 melt 후 이름) -> target_tables 컬럼명
      "성별": "gender",
      "지역": "region_name",
      "인구": "population",
      "통계연월": "base_date"   // 필요하면 이렇게 헤더 행에 있는 것도 매핑 가능
    },

    "column_roles_override": {
      "base_date": "metadata"   // target_tables.columns.role 를 덮어쓰고 싶을 때
    },

    "metadata_cells": [
      {
        "field": "base_date",   // target_tables.columns.name 중 metadata/both 로 설정된 필드
        "row": 1,               // 원본 df_raw 기준 0-based
        "col": 0,
        "parse_hint": "date_in_text"  // 선택: "date_in_text" 등, 사람이 보면 이해 가능한 힌트
      }
    ],

    "drop_unmapped_columns": true
  }
]
```

주의:

* file_name / sheet_name 은 반드시 **프로파일에 나온 문자열과 동일하게** 써야 합니다.
* 필요 없거나 통일이 안 되는 정보는:

  * target_tables 에 컬럼을 만들지 말고,
  * sheet_mappings 에서도 column_mapping 에 포함시키지 마십시오.
* CSV에 넣기에는 애매하거나 반복이 의미 없는 정보(제목, 통계 기준일, 단위 등)는:

  * target_tables.columns 에 role="metadata" 로 정의하고,
  * metadata_cells로 추출 규칙을 지정하세요.

### 3) 최종 응답 형식

반드시 아래 구조의 JSON **하나만** 출력하세요:

```json
{
  "target_tables": [ ... ],
  "sheet_mappings": [ ... ]
}
```

마크다운, 주석, 설명 문장 등은 JSON 바깥에 절대 쓰지 마십시오.
설명 텍스트(컬럼 description 등)는 한국어로 작성해도 됩니다.
"""

    prompt = f"""{dsl_spec}

아래는 이번에 처리해야 할 엑셀 시트들의 프로파일입니다:

<sheet_profiles>
{profiles_json}
</sheet_profiles>

위 정보를 바탕으로, 요구된 JSON 형식에 정확히 맞게 답변하세요.
"""
    return prompt


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다.")
    return text[start : end + 1]


def call_gemini_for_spec(
    api_key: str, sheet_profiles: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Gemini 3 Pro Preview 호출 → target_tables + sheet_mappings spec 반환
    """
    client = genai.Client(api_key=api_key)
    prompt = build_llm_prompt(sheet_profiles)

    resp = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
    )

    text = resp.text
    json_str = extract_json_object(text)
    spec = json.loads(json_str)
    return spec


# ======================================
# 3. 규칙 실행기 (Executor)
# ======================================


def apply_mapping_to_sheet(
    df_raw: pd.DataFrame,
    file_name: str,
    sheet_name: str,
    mapping: Dict[str, Any],
    table_def: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    하나의 (file, sheet)에 대해:
    - preprocess 규칙 적용
    - 필요 시 melt
    - column_mapping으로 canonical 컬럼 이름 부여
    - target_table 정의에 맞춰 canonical 컬럼만 유지
    - metadata_cells 로 메타데이터 추출

    반환:
    - df_canon: canonical 컬럼 + _source_file/_source_sheet
    - sheet_metadata: {"field_name": value, ...}
    """
    df = df_raw.copy()
    preprocess = mapping.get("preprocess", {})
    header_row = preprocess.get("header_row")
    drop_top = int(preprocess.get("drop_top_rows", 0) or 0)
    drop_bottom = int(preprocess.get("drop_bottom_rows", 0) or 0)
    drop_empty_rows = bool(preprocess.get("drop_empty_rows", True))
    drop_empty_cols = bool(preprocess.get("drop_empty_columns", True))
    drop_rows_matching = preprocess.get("drop_rows_matching", {})

    n_rows, n_cols = df.shape
    if header_row is None:
        header_row = 0
    header_row = max(0, min(int(header_row), n_rows - 1))

    # header_row부터 잘라서 헤더 적용
    df = df.iloc[header_row:, :].copy()
    if df.empty:
        return pd.DataFrame(), {}

    header = df.iloc[0].fillna("").astype(str).str.strip()
    df = df.iloc[1:, :].copy()
    df.columns = header

    # 추가 상단 행 제거
    if drop_top > 0 and drop_top < len(df):
        df = df.iloc[drop_top:, :].copy()

    # 하단 행 제거
    if drop_bottom > 0 and drop_bottom < len(df):
        df = df.iloc[:-drop_bottom, :].copy()

    # 빈 행/열 제거
    if drop_empty_rows:
        df = df.dropna(how="all")
    if drop_empty_cols:
        df = df.dropna(how="all", axis=1)
    df = df.reset_index(drop=True)

    # 특정 값(예: '합계')인 행 제거
    if drop_rows_matching:
        col_idx = drop_rows_matching.get("column_index")
        equals_vals = drop_rows_matching.get("equals", [])
        if col_idx is not None and equals_vals:
            col_idx = int(col_idx)
            if 0 <= col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                mask = df[col_name].astype(str).isin([str(v) for v in equals_vals])
                df = df[~mask].copy()
                df = df.reset_index(drop=True)

    # melt 단계 (wide → long 변환)
    melt_spec = mapping.get("melt")
    if melt_spec and melt_spec.get("enabled"):
        id_cols_spec = melt_spec.get("id_columns", [])
        id_cols = [c for c in id_cols_spec if c in df.columns]
        value_cols_spec = melt_spec.get("value_columns", "all_except_id")

        if isinstance(value_cols_spec, list):
            value_cols = [c for c in value_cols_spec if c in df.columns]
        else:
            value_cols = [c for c in df.columns if c not in id_cols]

        var_name = melt_spec.get("variable_name", "variable")
        val_name = melt_spec.get("value_name", "value")

        if value_cols:
            df = pd.melt(
                df,
                id_vars=id_cols,
                value_vars=value_cols,
                var_name=var_name,
                value_name=val_name,
            )

    # column_mapping 적용 → canonical 컬럼명으로 rename
    col_map = mapping.get("column_mapping", {})
    safe_map = {src: dst for src, dst in col_map.items() if src in df.columns}
    df = df.rename(columns=safe_map)

    # target_table 정의 기반 canonical 컬럼 만들기
    canonical_cols = [c["name"] for c in table_def.get("columns", [])]
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = None

    # canonical 컬럼만 유지
    df_canon = df[canonical_cols].copy()

    # 원본 출처 정보 추가 (CSV에는 항상 포함할 것)
    df_canon["_source_file"] = file_name
    df_canon["_source_sheet"] = sheet_name

    # metadata 셀 추출 (원본 df_raw 기준)
    metadata_cells = mapping.get("metadata_cells", [])
    sheet_meta_fields: Dict[str, Any] = {}
    for m in metadata_cells:
        field = m.get("field")
        row = m.get("row")
        col = m.get("col")
        if field is None or row is None or col is None:
            continue
        try:
            val = df_raw.iloc[int(row), int(col)]
        except Exception:
            val = None
        if pd.isna(val):
            val = None
        if val is not None:
            sheet_meta_fields[field] = str(val)

    return df_canon, sheet_meta_fields


def normalize_all_files(
    all_files: List[Dict[str, Any]], spec: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Any]]:
    """
    spec(target_tables + sheet_mappings)을 사용해서
    - 파일별/테이블별 canonical DataFrame 생성
    - metadata.json에 들어갈 summary 생성
    """
    target_tables = {t["name"]: t for t in spec.get("target_tables", [])}
    sheet_mappings = spec.get("sheet_mappings", [])

    # (file_name, sheet_name) -> [mappings]
    mapping_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for m in sheet_mappings:
        fname = m.get("file_name")
        sname = m.get("sheet_name")
        if not fname or not sname:
            continue
        mapping_index.setdefault((fname, sname), []).append(m)

    # 결과 구조
    outputs: Dict[str, Dict[str, pd.DataFrame]] = {}  # file -> table -> df
    tables_summary: Dict[str, Any] = {}  # table_name -> {columns, instances: [...]}

    for tname, tdef in target_tables.items():
        tables_summary[tname] = {
            "columns": tdef.get("columns", []),
            "instances": [],  # 각 파일/시트별 {file_name, sheet_name, row_count, metadata_fields}
        }

    for file_entry in all_files:
        fname = file_entry["file_name"]
        sheets: Dict[str, pd.DataFrame] = file_entry["sheets"]
        outputs[fname] = {}

        for sheet_name, df_raw in sheets.items():
            key = (fname, sheet_name)
            if key not in mapping_index:
                continue

            for mapping in mapping_index[key]:
                table_name = mapping.get("table_name")
                if table_name not in target_tables:
                    continue
                table_def = target_tables[table_name]

                df_canon, meta_fields = apply_mapping_to_sheet(
                    df_raw, fname, sheet_name, mapping, table_def
                )
                if df_canon.empty:
                    continue

                # outputs에 append
                if table_name not in outputs[fname]:
                    outputs[fname][table_name] = df_canon
                else:
                    outputs[fname][table_name] = pd.concat(
                        [outputs[fname][table_name], df_canon], ignore_index=True
                    )

                # metadata summary
                tables_summary[table_name]["instances"].append(
                    {
                        "file_name": fname,
                        "sheet_name": sheet_name,
                        "row_count": int(df_canon.shape[0]),
                        "metadata_fields": meta_fields,
                    }
                )

    # metadata.json의 상위 구조
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "llm_model": "gemini-3-pro-preview",
        "target_tables": spec.get("target_tables", []),
        "sheet_mappings": spec.get("sheet_mappings", []),
        "tables": tables_summary,
    }

    return outputs, metadata


# ======================================
# 4. Streamlit UI
# ======================================


def main():
    st.set_page_config(page_title="LLM 기반 Excel → 통일 CSV + 메타데이터", layout="wide")
    st.title("🧹 LLM 기반 Excel 정리기: 통일된 CSV + metadata.json")

    st.markdown(
        """
**목표**

* 엑셀 보고서(복잡한 헤더/주석/서식)를

  * 깔끔한 통일 스키마의 CSV로 만들고,
  * CSV로 넣기 애매한 정보는 metadata.json으로 분리합니다.
* 스키마/규칙은 사람이 하드코딩하지 않고, **LLM(Gemini 3 Pro Preview)가 스스로 설계**합니다.
* 파이썬 코드는 그 스펙을 그대로 실행하는 **일반화된 실행기**입니다.
        """
    )

    st.sidebar.header("Gemini 설정")
    api_key_input = st.sidebar.text_input(
        "GEMINI_API_KEY",
        type="password",
        help="Google AI Studio / Gemini API 키. 환경변수 GEMINI_API_KEY로도 설정 가능.",
    )
    api_key_env = os.environ.get("GEMINI_API_KEY")
    api_key = api_key_input or api_key_env

    if not api_key:
        st.sidebar.warning("LLM 기능을 쓰려면 GEMINI_API_KEY가 필요합니다.")

    uploaded_files = st.file_uploader(
        "엑셀 파일 업로드 (.xls, .xlsx) — 여러 개 가능",
        type=["xls", "xlsx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("먼저 엑셀 파일을 하나 이상 업로드해 주세요.")
        return

    # 1) 엑셀 로드 + 시트 프로파일 생성
    all_files: List[Dict[str, Any]] = []
    all_sheet_profiles: List[Dict[str, Any]] = []

    st.subheader("1️⃣ 업로드된 파일 & 시트 구조 확인")

    for f in uploaded_files:
        st.markdown(f"#### 📁 {f.name}")
        try:
            sheets = read_all_sheets(f)
        except Exception as e:
            st.error(f"{f.name} 읽기 실패: {e}")
            continue

        file_entry = {"file_name": f.name, "sheets": sheets}
        all_files.append(file_entry)

        st.write("시트 목록:", list(sheets.keys()))

        for sheet_name, df_raw in sheets.items():
            prof = profile_sheet_for_llm(df_raw, f.name, sheet_name)
            all_sheet_profiles.append(prof)

            with st.expander(f"시트 프로파일: {sheet_name}"):
                st.json(prof)

    # 2) LLM으로 스키마/규칙 생성
    st.subheader("2️⃣ LLM으로 통일 스키마 + 시트별 변환 규칙 생성")

    spec_state_key = "llm_transform_spec"
    spec_container = st.empty()

    if st.button("Gemini 3 Pro Preview로 스키마/규칙 설계 요청"):
        if not api_key:
            st.error("GEMINI_API_KEY가 필요합니다.")
        else:
            with st.spinner("Gemini 3 Pro Preview에게 설계 요청 중..."):
                try:
                    spec = call_gemini_for_spec(api_key, all_sheet_profiles)
                    st.session_state[spec_state_key] = spec
                    st.success("LLM transform spec 생성 완료!")
                except Exception as e:
                    st.error(f"LLM 호출/파싱 실패: {e}")

    spec = st.session_state.get(spec_state_key)
    if spec:
        spec_container.subheader("LLM transform spec (요약)")
        spec_container.code(
            json.dumps(spec, ensure_ascii=False, indent=2)[:6000], language="json"
        )
    else:
        spec_container.info("아직 LLM transform spec이 없습니다. 위 버튼으로 생성해 주세요.")

    # 3) 규칙 실행 → CSV + metadata.json
    st.subheader("3️⃣ 규칙 실행 → 통일된 CSV + metadata.json 생성")

    if st.button("규칙 실행 및 결과 생성"):
        if not spec:
            st.error("먼저 LLM으로부터 transform spec을 생성해야 합니다.")
        else:
            with st.spinner("규칙 실행 중..."):
                outputs, metadata = normalize_all_files(all_files, spec)

                st.success("변환 완료! 아래에서 CSV와 metadata.json을 확인/다운로드할 수 있습니다.")

                # 테이블 정의에서 role 에 따라 어떤 컬럼이 data/metadata 인지 확인
                target_tables = {t["name"]: t for t in spec.get("target_tables", [])}
                roles_by_table: Dict[str, Dict[str, str]] = {}
                for tname, tdef in target_tables.items():
                    roles = {}
                    for col in tdef.get("columns", []):
                        roles[col["name"]] = col.get("role", "data")
                    roles_by_table[tname] = roles

                # 파일별/테이블별 CSV 다운로드 버튼
                for file_entry in all_files:
                    fname = file_entry["file_name"]
                    if fname not in outputs:
                        continue
                    st.markdown(f"### 📁 {fname} — 변환된 테이블")

                    for table_name, df_canon in outputs[fname].items():
                        st.markdown(f"#### 📊 테이블: `{table_name}`")

                        # role 기반으로 CSV에 포함할 컬럼 결정
                        roles = roles_by_table.get(table_name, {})
                        data_cols = [
                            c
                            for c in df_canon.columns
                            if c in roles and roles.get(c, "data") in ("data", "both")
                        ]
                        # 항상 포함할 출처 컬럼
                        extra_cols = ["_source_file", "_source_sheet"]
                        csv_cols = data_cols + [c for c in extra_cols if c in df_canon.columns]

                        if not csv_cols:
                            st.info("이 테이블에서 CSV에 포함할 데이터 컬럼이 없습니다.")
                            continue

                        df_csv = df_canon[csv_cols].copy()
                        st.dataframe(df_csv.head(20))

                        csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")
                        safe_fname = os.path.splitext(os.path.basename(fname))[0]
                        out_name = f"{safe_fname}__{table_name}.csv"

                        st.download_button(
                            label=f"⬇️ {out_name} 다운로드",
                            data=csv_bytes,
                            file_name=out_name,
                            mime="text/csv",
                        )

                # metadata.json 다운로드
                st.markdown("### 🧾 metadata.json")
                st.code(json.dumps(metadata, ensure_ascii=False, indent=2)[:6000], language="json")
                meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label="⬇️ metadata.json 다운로드",
                    data=meta_bytes,
                    file_name="metadata.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
