import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide"
)

# ---------- 커스텀 CSS ----------
st.markdown("""
<style>
    .title-container {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .section-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 12px;
        margin: 25px 0 15px 0;
        font-size: 20px;
        font-weight: 700;
    }

    .info-box {
        background-color: #fff8e6;
        border-left: 5px solid #ffc107;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
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
    <h1>🔍 데이터 탐색하기</h1>
    <p style="font-size:18px; color:#6b7a99;">여러 각도로 데이터를 살펴보며 뇌졸중과의 관계를 찾아봅시다 🧐</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 1. 나이와 평균 혈당 분포 (히스토그램)
# ==========================================================
st.markdown('<div class="section-header">📊 나이와 평균 혈당의 분포</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df, x="age", nbins=40,
        title="나이(age) 분포",
        color_discrete_sequence=["#4b6cb7"]
    )
    fig_age_hist.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.05
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level", nbins=40,
        title="평균 혈당(avg_glucose_level) 분포",
        color_discrete_sequence=["#e07a5f"]
    )
    fig_glucose_hist.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.05
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

with st.expander("📖 해석 보기"):
    st.write(f"""
    - **나이 분포**: 전체 사람 수는 {len(df):,}명이며, 나이는 0세부터 최대
      {df['age'].max():.0f}세까지 분포합니다. 히스토그램의 막대가 높을수록
      해당 나이대의 사람이 많다는 뜻입니다.
    - **평균 혈당 분포**: 평균 혈당 값은 대체로 낮은 구간(정상 범위)에
      많이 몰려있고, 오른쪽으로 길게 꼬리를 그리는 모양(치우친 분포)을
      보입니다. 이는 혈당이 매우 높은 소수의 사람들이 존재한다는 뜻입니다.
    - 이렇게 분포 모양을 먼저 확인하면, 이후 분석에서 평균이나 중앙값 중
      어떤 값을 대표값으로 쓸지 판단하는 데 도움이 됩니다.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 2. 뇌졸중 여부에 따른 나이/혈당 상자그림 + 평균 표
# ==========================================================
st.markdown('<div class="section-header">📦 뇌졸중 여부에 따른 나이·혈당 비교</div>', unsafe_allow_html=True)

df_box = df.copy()
df_box["stroke_label"] = df_box["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df_box, x="stroke_label", y="age",
        color="stroke_label",
        title="뇌졸중 여부에 따른 나이 비교",
        color_discrete_map={"뇌졸중 없음": "#4b6cb7", "뇌졸중 있음": "#e07a5f"}
    )
    fig_age_box.update_layout(xaxis_title="", yaxis_title="나이", showlegend=False)
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_glucose_box = px.box(
        df_box, x="stroke_label", y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        color_discrete_map={"뇌졸중 없음": "#4b6cb7", "뇌졸중 있음": "#e07a5f"}
    )
    fig_glucose_box.update_layout(xaxis_title="", yaxis_title="평균 혈당", showlegend=False)
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 평균값 표
mean_table = df_box.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
mean_table = mean_table.reset_index().rename(columns={"stroke_label": "구분"})

st.write("**📋 그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

with st.expander("📖 해석 보기"):
    age_diff = mean_table.loc[mean_table["구분"] == "뇌졸중 있음", "나이 평균"].values[0] - \
               mean_table.loc[mean_table["구분"] == "뇌졸중 없음", "나이 평균"].values[0]
    glucose_diff = mean_table.loc[mean_table["구분"] == "뇌졸중 있음", "평균 혈당 평균"].values[0] - \
                   mean_table.loc[mean_table["구분"] == "뇌졸중 없음", "평균 혈당 평균"].values[0]
    st.write(f"""
    - **상자그림 읽는 법**: 상자의 가운데 선은 중앙값, 상자의 위아래 끝은
      각각 3사분위수와 1사분위수를 나타냅니다. 상자 밖의 점들은 이상치입니다.
    - **나이 비교**: 뇌졸중을 겪은 사람들의 평균 나이가 그렇지 않은
      사람들보다 약 **{age_diff:.1f}세 더 많습니다**. 상자 자체의 위치도
      더 높은 나이대에 있는 것을 확인할 수 있습니다.
    - **평균 혈당 비교**: 뇌졸중을 겪은 사람들의 평균 혈당이 그렇지 않은
      사람들보다 약 **{glucose_diff:.1f} 더 높습니다**.
    - 즉, 나이가 많고 혈당이 높을수록 뇌졸중과 관련이 있을 가능성을
      시각적으로 확인할 수 있습니다. (다만 상관관계가 곧 인과관계는 아니라는
      점에 유의해야 합니다!)
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 3. 고혈압/심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# ==========================================================
st.markdown('<div class="section-header">📈 고혈압·심장병 유무에 따른 뇌졸중 비율</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyper_ratio["stroke"] = hyper_ratio["stroke"] * 100

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension", y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율(%)",
        color="hypertension",
        color_discrete_map={"고혈압 없음": "#4b6cb7", "고혈압 있음": "#e07a5f"},
        text_auto=".2f"
    )
    fig_hyper.update_layout(xaxis_title="", yaxis_title="뇌졸중 비율(%)", showlegend=False)
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100

    fig_heart = px.bar(
        heart_ratio, x="heart_disease", y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율(%)",
        color="heart_disease",
        color_discrete_map={"심장병 없음": "#4b6cb7", "심장병 있음": "#e07a5f"},
        text_auto=".2f"
    )
    fig_heart.update_layout(xaxis_title="", yaxis_title="뇌졸중 비율(%)", showlegend=False)
    st.plotly_chart(fig_heart, use_container_width=True)

with st.expander("📖 해석 보기"):
    h0 = hyper_ratio.loc[hyper_ratio["hypertension"] == "고혈압 없음", "stroke"].values[0]
    h1 = hyper_ratio.loc[hyper_ratio["hypertension"] == "고혈압 있음", "stroke"].values[0]
    d0 = heart_ratio.loc[heart_ratio["heart_disease"] == "심장병 없음", "stroke"].values[0]
    d1 = heart_ratio.loc[heart_ratio["heart_disease"] == "심장병 있음", "stroke"].values[0]
    st.write(f"""
    - **고혈압**: 고혈압이 없는 사람의 뇌졸중 비율은 **{h0:.2f}%**인 반면,
      고혈압이 있는 사람의 뇌졸중 비율은 **{h1:.2f}%**로 약
      **{h1/h0:.1f}배** 더 높습니다.
    - **심장병**: 심장병이 없는 사람의 뇌졸중 비율은 **{d0:.2f}%**인 반면,
      심장병이 있는 사람의 뇌졸중 비율은 **{d1:.2f}%**로 약
      **{d1/d0:.1f}배** 더 높습니다.
    - 두 질환 모두 뇌졸중 발생과 뚜렷한 관련이 있어 보입니다. 이런 기존
      질환이 있는 사람들은 특히 건강 관리에 신경 써야 함을 알 수 있습니다.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 4. bmi 결측치와 뇌졸중 비율
# ==========================================================
st.markdown('<div class="section-header">🩺 체질량지수(bmi) 결측치와 뇌졸중 비율</div>', unsafe_allow_html=True)

bmi_missing = df[df["bmi"].isnull()]
bmi_missing_count = len(bmi_missing)
bmi_missing_stroke_ratio = bmi_missing["stroke"].mean() * 100 if bmi_missing_count > 0 else 0
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi가 비어 있는 사람들", "전체 사람들"],
    "인원 수": [bmi_missing_count, len(df)],
    "뇌졸중 비율(%)": [round(bmi_missing_stroke_ratio, 2), round(overall_stroke_ratio, 2)]
})

st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

with st.expander("📖 해석 보기"):
    st.write(f"""
    - bmi 값이 비어 있는 사람은 총 **{bmi_missing_count}명**입니다.
    - 이 사람들의 뇌졸중 비율은 **{bmi_missing_stroke_ratio:.2f}%**로,
      전체 뇌졸중 비율인 **{overall_stroke_ratio:.2f}%**와 비교했을 때
      {"더 높습니다" if bmi_missing_stroke_ratio > overall_stroke_ratio else "더 낮거나 비슷합니다"}.
    - 만약 두 비율의 차이가 크다면, bmi 결측치가 단순한 우연이 아니라
      특정 집단(예: 건강검진을 잘 받지 않는 사람들)에서 더 많이 발생했을
      가능성을 의심해볼 수 있습니다. 이런 경우 결측치를 무작정 삭제하면
      분석 결과가 왜곡될 수 있으므로 주의가 필요합니다.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 5. 흡연 상태별 사람 수
# ==========================================================
st.markdown('<div class="section-header">🚬 흡연 상태(smoking_status)별 사람 수</div>', unsafe_allow_html=True)

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True, hide_index=True)

with st.expander("📖 해석 보기"):
    st.write(f"""
    - 흡연 상태는 총 **{df['smoking_status'].nunique()}가지** 값으로
      구분되어 있습니다.
    - 각 범주별 인원 수를 확인하면, 특정 범주(예: 정보가 없는
      'Unknown')에 사람이 많이 몰려있는지, 혹은 골고루 분포되어 있는지
      알 수 있습니다.
    - 흡연 여부에 따른 뇌졸중 비율을 추가로 분석해보면, 흡연이 뇌졸중에
      미치는 영향도 살펴볼 수 있습니다. (다음 탐색 단계에서 시도해봐도
      좋습니다!)
    """)
