# 🧩 AI 통합 데이터 정제기 (Many to One)

**"AI Integrated Data Cleaner: From Chaos to Order"**

이 프로젝트는 서로 다른 형식(Schema)을 가진 여러 비정형 엑셀 파일들을 **Google Gemini AI**의 코딩 능력을 빌려 **단 하나의 통일된 CSV 파일**로 통합하는 도구입니다.

## 💡 기획 의도

현업에서 데이터 분석가가 겪는 가장 큰 고통은 **"데이터 전처리(Data Preprocessing)"**입니다. 특히 여러 지사나 부서에서 취합된 엑셀 파일들은 파일명만 같을 뿐, 내부 구조는 제각각인 경우가 많습니다.

*   A 파일: 헤더가 1행에 있고, 날짜가 '2024-01-01' 형식.
*   B 파일: 헤더가 3행에 있고, 날짜가 '2024년 1월 1일' 형식이며, 중간에 '소계' 행이 섞여 있음.
*   C 파일: 가로로 긴(Wide) 형태라 세로로(Long) 바꿔야 함.

기존에는 이를 해결하기 위해 사람이 일일이 엑셀을 열어보고, `if file == 'A': ... elif file == 'B': ...` 식의 하드코딩된 스크립트를 짰습니다. 하지만 파일이 수십 개라면? 혹은 매달 양식이 조금씩 바뀐다면? 이 방식은 유지보수가 불가능합니다.

**"AI가 코드를 짜주면 어떨까?"**

이 프로그램은 **LLM(Large Language Model)에게 "데이터를 보고, 정해진 목표 형태(Target Schema)로 바꾸는 파이썬 코드를 작성해줘"라고 시키는 방식**으로 이 문제를 해결합니다. 규칙(Rule)을 만드는 것이 아니라, 실행 가능한 코드(Code)를 생성함으로써 무한한 유연성을 확보했습니다.

## 🚀 주요 기능 (Workflow)

1.  **Any Excel Input**: 형식이 제각각인 엑셀 파일들을 한 번에 업로드합니다.
2.  **AI Schema Proposal**: AI가 업로드된 파일들의 샘플을 분석하여, 공통적으로 뽑아낼 수 있는 핵심 컬럼(예: 지역, 날짜, 인구수)을 스스로 제안합니다.
3.  **AI Code Generation**: 각 파일별로 맞춤형 전처리 파이썬 코드를 AI가 자동으로 작성합니다. (Pandas 활용)
    *   복잡한 헤더 처리
    *   불필요한 행 삭제
    *   Melt (Pivot 해제)
    *   날짜 포맷 통일 등
4.  **Interactive Execution**: 사용자는 AI가 짠 코드를 눈으로 확인하고, 즉석에서 실행해보고, 필요하면 수정할 수 있습니다. (Human-in-the-loop)
5.  **One-Click Merge**: 모든 파일이 성공적으로 변환되면, 버튼 하나로 통합된 `merged_data.csv`를 다운로드합니다.

## 🛠 기술 스택

*   **Frontend/App**: Streamlit (Python)
*   **AI Engine**: gemini-3-pro-preview
*   **Data Processing**: Pandas, OpenPyXL
*   **Interface**: Google Gen AI SDK

## 📝 사용 방법

1.  `GEMINI_API_KEY`를 입력합니다.
2.  엑셀 파일(.xlsx, .xls)들을 드래그 앤 드롭으로 업로드합니다.
3.  **"AI 스키마 자동 제안"** 버튼을 눌러 목표 컬럼을 정의합니다.
4.  각 파일 탭에서 **"코드 생성"** -> **"실행"** 버튼을 차례로 누릅니다.
5.  모든 파일의 변환이 끝나면 하단의 **"통합 CSV 다운로드"**를 클릭합니다.

