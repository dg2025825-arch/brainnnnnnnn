import streamlit as st
import pandas as pd

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------- 데이터 불러오기 ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------- 제목 ----------
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("### 데이터를 통해 뇌졸중을 예측해보는 실습 공간입니다.")
st.divider()

# ---------- 데이터 소개 ----------
st.header("📌 데이터 소개")
st.write(
    """
    이 실습실에서는 환자들의 다양한 건강 정보를 담은 데이터를 활용해서
    뇌졸중(stroke) 발생 여부를 예측해보는 학습을 진행합니다.
    아래에서 데이터의 전체적인 모습을 먼저 살펴봅시다.
    """
)

# ---------- 숫자 카드 4개 ----------
total_people = len(df)
total_columns = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# ---------- 열 설명 표 ----------
st.header("📋 열(컬럼) 설명")
st.write("아래 표의 **'우리말 뜻'** 칸은 비어 있습니다. 교재를 참고하여 직접 채워보세요!")

# 열별 값의 종류(고유값 개수)와 결측치 개수 계산
column_info = []
for col in df.columns:
    unique_count = df[col].nunique(dropna=True)
    null_count = df[col].isnull().sum()
    column_info.append({
        "열 이름": col,
        "우리말 뜻": "",   # 학생이 직접 채우는 빈 칸
        "값의 종류 (고유값 개수)": unique_count,
        "빈 값 개수": null_count
    })

info_df = pd.DataFrame(column_info)

st.data_editor(
    info_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    key="column_info_editor"
)

st.divider()

# ---------- 데이터 미리보기 ----------
st.header("👀 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ---------- 데이터 출처 ----------
st.header("📚 데이터 출처")
st.info("여기에 교재에 나온 데이터 출처를 직접 적어보세요.")

source_text = st.text_area(
    "출처를 입력하세요:",
    placeholder="예) 교재 몇 쪽을 참고하여 출처를 작성해보세요.",
    height=100
)

if source_text:
    st.success("✅ 작성한 출처가 저장되었습니다 (이 화면 안에서만 유지됩니다).")
    st.write(source_text)
