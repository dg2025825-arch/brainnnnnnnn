import streamlit as st
import pandas as pd

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ---------- 커스텀 CSS ----------
st.markdown("""
<style>
    /* 전체 폰트 및 배경 살짝 조정 */
    .main {
        background-color: #fafbfd;
    }

    /* 제목 스타일 */
    .title-container {
        text-align: center;
        padding: 30px 0 10px 0;
    }

    /* metric 카드 꾸미기 */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f6f8ff 0%, #e9f0ff 100%);
        border: 1px solid #d6e2ff;
        border-radius: 16px;
        padding: 20px 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        text-align: center;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        font-weight: 600;
        color: #5b6b8c;
    }

    div[data-testid="stMetricValue"] {
        font-size: 30px;
        font-weight: 800;
        color: #2b4c7e;
    }

    /* 섹션 헤더 스타일 */
    .section-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 25px 0 15px 0;
        font-size: 20px;
        font-weight: 700;
    }

    /* 안내 박스 */
    .info-box {
        background-color: #fff8e6;
        border-left: 5px solid #ffc107;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* 출처 박스 */
    .source-box {
        background-color: #eafaf1;
        border-left: 5px solid #27ae60;
        padding: 15px 20px;
        border-radius: 8px;
    }

    hr {
        border: none;
        border-top: 2px dashed #cdd6e0;
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 데이터 불러오기 ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ---------- 제목 ----------
st.markdown("""
<div class="title-container">
    <h1>🧠 뇌졸중 예측 실습실 🩺</h1>
    <p style="font-size:18px; color:#6b7a99;">데이터를 통해 뇌졸중을 예측해보는 실습 공간입니다 ✨</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ---------- 데이터 소개 ----------
st.markdown('<div class="section-header">📌 데이터 소개</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
이 실습실에서는 환자들의 다양한 건강 정보를 담은 데이터를 활용해서
🧠 <b>뇌졸중(stroke)</b> 발생 여부를 예측해보는 학습을 진행합니다.<br>
아래에서 데이터의 전체적인 모습을 먼저 살펴봅시다! 🔍
</div>
""", unsafe_allow_html=True)

# ---------- 숫자 카드 4개 ----------
total_people = len(df)
total_columns = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_people * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="📊 열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="🚨 뇌졸중 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="📈 뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

# ---------- 열 설명 표 ----------
st.markdown('<div class="section-header">📋 열(컬럼) 설명</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
✏️ 아래 표의 <b>'우리말 뜻'</b> 칸은 비어 있습니다. 교재를 참고하여 직접 채워보세요!
</div>
""", unsafe_allow_html=True)

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

# ---------- 데이터 미리보기 ----------
st.markdown('<div class="section-header">👀 데이터 미리보기 (처음 5줄)</div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)

# ---------- 데이터 출처 ----------
st.markdown('<div class="section-header">📚 데이터 출처</div>', unsafe_allow_html=True)

source_text = st.text_area(
    "✍️ 교재를 참고해서 출처를 입력하세요:",
    placeholder="예) 교재 몇 쪽을 참고하여 출처를 작성해보세요.",
    height=100
)

if source_text:
    st.markdown(f"""
    <div class="source-box">
    ✅ <b>작성한 출처</b><br>{source_text}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
    ⏳ 아직 출처가 입력되지 않았어요. 교재를 참고해서 위 칸에 적어보세요!
    </div>
    """, unsafe_allow_html=True)
