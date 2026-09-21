import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# ---------- 기본 설정 ----------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
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
    .warning-box {
        background-color: #fdecea;
        border-left: 5px solid #e74c3c;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
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
        font-size: 28px;
        font-weight: 800;
        color: #2b4c7e;
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
    <h1>🤖 뇌졸중 분류 모델 만들기</h1>
    <p style="font-size:18px; color:#6b7a99;">여러 속성을 이용해 뇌졸중을 예측하는 모델을 만들고 비교해봅시다 🔬</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 우리말 이름 매핑
# ==========================================================
kor_name = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
eng_name = {v: k for k, v in kor_name.items()}

feature_candidates = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외

# ==========================================================
# 속성 선택
# ==========================================================
st.markdown('<div class="section-header">🎛️ 입력 속성 선택하기</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
아래에서 뇌졸중 예측에 사용할 속성을 골라보세요. 최소 <b>2개 이상</b> 선택해야 합니다.
</div>
""", unsafe_allow_html=True)

selected_kor = st.multiselect(
    "사용할 속성을 선택하세요:",
    options=[kor_name[c] for c in feature_candidates],
    default=[kor_name[c] for c in default_features]
)

selected_features = [eng_name[k] for k in selected_kor]

if len(selected_features) < 2:
    st.markdown("""
    <div class="warning-box">
    ⚠️ 속성을 <b>2개 이상</b> 선택해야 모델을 만들 수 있습니다. 위에서 속성을 더 선택해주세요.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 데이터 준비
# ==========================================================
work_df = df.copy().sort_values("id").reset_index(drop=True)

use_bmi = "bmi" in selected_features

# 10명씩 묶어 앞 3명을 테스트용으로 고정
row_number = np.arange(len(work_df))
is_test = (row_number % 10) < 3

test_df = work_df[is_test].copy()
train_df = work_df[~is_test].copy()

# bmi 결측치는 훈련용 중앙값으로 채우기 (선택된 경우에만)
if use_bmi:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

# 열 순서를 selected_features 순서로 명확히 고정
X_train = train_df[selected_features].reset_index(drop=True)
y_train = train_df["stroke"].reset_index(drop=True)
X_test = test_df[selected_features].reset_index(drop=True)
y_test = test_df["stroke"].reset_index(drop=True)

st.markdown('<div class="section-header">📦 데이터 나누기 결과</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("전체 사람 수", f"{len(work_df):,} 명")
with col_b:
    st.metric("학습용 사람 수", f"{len(train_df):,} 명")
with col_c:
    st.metric("테스트용 사람 수", f"{len(test_df):,} 명")

with st.expander("📖 데이터를 나눈 방법 설명 보기"):
    st.write(f"""
    - 전체 사람을 **id 순서대로 정렬**한 뒤, **10명씩 한 묶음**으로 나눴습니다.
    - 각 묶음에서 **앞 3명은 테스트용**으로, **나머지 7명은 학습용**으로 사용했습니다.
    - 이렇게 나누면 전체 {len(work_df):,}명 중 테스트용은 **{len(test_df):,}명**이 됩니다.
    - 체질량지수(bmi)를 선택한 경우, 비어 있는 값은 **학습용 데이터의 중앙값**으로
      채웠습니다. (테스트용 데이터를 미리 들여다보면 안 되기 때문에, 반드시 학습용
      기준으로 채워야 공정한 평가가 됩니다.)
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 모델 학습
# ==========================================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 정확도 계산
log_train_acc = accuracy_score(y_train, log_model.predict(X_train))
log_test_acc = accuracy_score(y_test, log_model.predict(X_test))

tree_train_acc = accuracy_score(y_train, tree_model.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree_model.predict(X_test))

dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test))

# ==========================================================
# 정확도 카드
# ==========================================================
st.markdown('<div class="section-header">🎯 모델별 정확도 비교</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**로지스틱 회귀 (확률로 답하는 모델)**")
    st.metric("테스트 정확도", f"{log_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {log_train_acc*100:.2f}% ・ 테스트 정확도: {log_test_acc*100:.2f}%")

with col2:
    st.markdown("**의사결정트리 (질문으로 답하는 모델)**")
    st.metric("테스트 정확도", f"{tree_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {tree_train_acc*100:.2f}% ・ 테스트 정확도: {tree_test_acc*100:.2f}%")

with col3:
    st.markdown("**기준 모델 (무조건 다수결로 답하는 모델)**")
    st.metric("테스트 정확도", f"{dummy_test_acc*100:.2f} %")
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.2f}% ・ 테스트 정확도: {dummy_test_acc*100:.2f}%")

with st.expander("📖 해석 보기 (정확도가 높다고 다 좋은 모델일까?)"):
    st.write(f"""
    - **기준 모델**은 입력값을 전혀 보지 않고, 학습용 데이터에서 가장 많았던
      답(대부분 "뇌졸중 아님")으로만 찍는 아주 단순한 모델입니다. 그런데도
      정확도가 **{dummy_test_acc*100:.2f}%**로 꽤 높게 나옵니다.
    - 그 이유는 전체 데이터에서 뇌졸중인 사람의 비율이 매우 낮기 때문입니다.
      "아니오"라고만 답해도 대부분 맞기 때문에 정확도가 높게 나오는 것이죠.
    - 따라서 로지스틱 회귀나 의사결정트리 모델의 정확도가 기준 모델보다
      **얼마나 더 높은지**를 함께 봐야 진짜 성능을 판단할 수 있습니다.
    - 만약 두 모델의 정확도가 기준 모델과 비슷하다면, 모델이 실제로는
      뇌졸중 환자를 잘 찾아내지 못하고 있을 가능성이 큽니다. (정확도 외에
      재현율 같은 다른 지표도 함께 살펴봐야 하는 이유입니다.)
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 산점도 + 결정경계
# ==========================================================
st.markdown('<div class="section-header">🗺️ 결정 경계 시각화</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
고른 속성 중 두 가지를 골라 가로축과 세로축으로 삼아, 모델이 어떻게 구분하는지 그림으로 살펴봅시다.
</div>
""", unsafe_allow_html=True)

col_x, col_y = st.columns(2)
with col_x:
    x_axis_kor = st.selectbox("가로축으로 사용할 속성:", options=selected_kor, index=0)
with col_y:
    remaining = [k for k in selected_kor if k != x_axis_kor]
    y_axis_kor = st.selectbox("세로축으로 사용할 속성:", options=remaining, index=0)

x_axis = eng_name[x_axis_kor]
y_axis = eng_name[y_axis_kor]

other_features = [f for f in selected_features if f not in [x_axis, y_axis]]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for f in other_features:
    fixed_values[f] = X_test[f].median()

# 산점도 그리기
plot_df = test_df.copy()
plot_df["뇌졸중 여부"] = plot_df["stroke"].map({0: "뇌졸중 아님", 1: "뇌졸중"})

fig = go.Figure()

# 의사결정트리 배경 칠하기 (결정 영역)
x_min, x_max = X_test[x_axis].min(), X_test[x_axis].max()
y_min, y_max = X_test[y_axis].min(), X_test[y_axis].max()
x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1
y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1

xx, yy = np.meshgrid(
    np.linspace(x_min - x_pad, x_max + x_pad, 200),
    np.linspace(y_min - y_pad, y_max + y_pad, 200)
)

grid_df = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]
grid_df = grid_df[selected_features]  # 학습 시 사용한 열 순서 맞추기

tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig.add_trace(go.Contour(
    x=np.linspace(x_min - x_pad, x_max + x_pad, 200),
    y=np.linspace(y_min - y_pad, y_max + y_pad, 200),
    z=tree_pred_grid,
    showscale=False,
    colorscale=[[0, "rgba(75,108,183,0.15)"], [1, "rgba(224,122,95,0.15)"]],
    contours=dict(coloring="fill"),
    line=dict(width=0),
    hoverinfo="skip",
    name="의사결정트리 영역"
))

# 실제 산점도
colors = {"뇌졸중 아님": "#4b6cb7", "뇌졸중": "#e74c3c"}
for label, color in colors.items():
    sub = plot_df[plot_df["뇌졸중 여부"] == label]
    fig.add_trace(go.Scatter(
        x=sub[x_axis], y=sub[y_axis],
        mode="markers",
        marker=dict(color=color, size=8, opacity=0.7, line=dict(width=0.5, color="white")),
        name=label
    ))

# 로지스틱 회귀 결정 경계선 (0.5 기준)
coef_dict = dict(zip(selected_features, log_model.coef_[0]))
intercept = log_model.intercept_[0]

w_x = coef_dict[x_axis]
w_y = coef_dict[y_axis]

# 다른 속성들의 고정값이 만드는 상수항 더하기
fixed_sum = sum(coef_dict[f] * fixed_values[f] for f in other_features)
c = intercept + fixed_sum

line_drawn = False
line_out_of_range = False

if abs(w_y) > 1e-12:
    x_line = np.linspace(x_min - x_pad, x_max + x_pad, 200)
    y_line = -(w_x * x_line + c) / w_y

    # 선이 그림 범위 안에 있는지 확인
    y_plot_min, y_plot_max = y_min - y_pad, y_max + y_pad
    in_range_mask = (y_line >= y_plot_min) & (y_line <= y_plot_max)

    if in_range_mask.any():
        fig.add_trace(go.Scatter(
            x=x_line[in_range_mask], y=y_line[in_range_mask],
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="로지스틱 회귀 경계선 (0.5)"
        ))
        line_drawn = True
        if not in_range_mask.all():
            line_out_of_range = True  # 일부만 보임
    else:
        line_out_of_range = True
else:
    line_out_of_range = True

fig.update_layout(
    title=f"{x_axis_kor} vs {y_axis_kor} 산점도 및 결정 경계",
    xaxis_title=x_axis_kor,
    yaxis_title=y_axis_kor,
    legend_title="실제 뇌졸중 여부",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 고정값 설명 글
if other_features:
    fixed_text = ", ".join([f"{kor_name[f]} = {fixed_values[f]:.2f}" for f in other_features])
    st.info(f"📌 그림에 나타나지 않은 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.info("📌 선택한 속성이 두 개뿐이라 고정한 속성은 없습니다.")

if not line_drawn:
    st.warning("⚠️ 로지스틱 회귀의 경계선(0.5 기준선)이 이 그림의 범위 밖에 있어 화면에 표시되지 않았습니다.")
elif line_out_of_range:
    st.warning("⚠️ 로지스틱 회귀의 경계선 일부가 그림 범위 밖으로 벗어나 있습니다.")

with st.expander("📖 해석 보기 (그림을 어떻게 읽나요?)"):
    st.write(f"""
    - **점**: 실제 테스트 데이터의 사람들입니다. 파란 점은 실제로 뇌졸중이
      아니었던 사람, 빨간 점은 실제로 뇌졸중이었던 사람입니다.
    - **옅은 색 배경**: 의사결정트리 모델이 이 위치에 있는 사람을
      "뇌졸중이다/아니다"로 나눈 영역입니다. 배경색이 바뀌는 경계가 바로
      트리 모델이 그은 구분선입니다.
    - **검은 점선**: 로지스틱 회귀 모델이 "50% 확률"로 판단을 뒤집는
      경계선입니다. 이 선을 기준으로 한쪽은 "뇌졸중일 확률이 더 높다",
      다른 한쪽은 "아닐 확률이 더 높다"고 판단합니다.
    - 실제로 빨간 점(뇌졸중 있음)이 대부분 나이가 많고 혈당이 높은 쪽에
      몰려있다면, 두 모델이 그 방향으로 경계를 그었는지 확인해보세요.
    - 두 축으로 사용하지 않은 다른 속성들은 모두 특정 값(중앙값)으로
      고정해두고 계산한 것이므로, 실제 데이터와 완벽히 일치하지 않을 수
      있다는 점에 유의하세요.
    """)

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# 트리 구조 시각화 (graphviz DOT)
# ==========================================================
st.markdown('<div class="section-header">🌳 의사결정트리 구조 보기</div>', unsafe_allow_html=True)

tree = tree_model.tree_
feature_names = selected_features
tree_classes = tree_model.classes_  # 예: [0, 1] 순서 확인용

def get_node_info(node_id):
    """
    해당 노드에 도달한 훈련 데이터 수, 뇌졸중(1)인 사람 수, 비율을 계산.
    tree.value는 [해당 노드의 클래스별 가중 인원 수] 형태이며,
    class_weight를 안 줬으면 실제 인원 수와 같습니다.
    tree_model.classes_ 순서에 맞춰 1(뇌졸중)의 위치를 찾습니다.
    """
    value = tree.value[node_id][0]  # 예: [클래스0 개수, 클래스1 개수]
    n_total = int(tree.n_node_samples[node_id])

    # classes_ 배열에서 1(뇌졸중)이 몇 번째 위치인지 찾기
    class_list = list(tree_classes)
    if 1 in class_list:
        idx_yes = class_list.index(1)
        n_yes = int(round(value[idx_yes]))
    else:
        n_yes = 0

    ratio = n_yes / n_total if n_total > 0 else 0
    return n_total, n_yes, ratio

def build_dot(node_id=0):
    lines = []
    is_leaf = (tree.children_left[node_id] == -1) and (tree.children_right[node_id] == -1)
    n_total, n_yes, ratio = get_node_info(node_id)

    if is_leaf:
        pred = 1 if ratio >= 0.5 else 0
        label_text = "뇌졸중" if pred == 1 else "정상"
        fill_color = "#f9c2c2" if pred == 1 else "#c2d4f9"
        node_label = f"{label_text}\\n인원 {n_total}명 (뇌졸중 {n_yes}명)\\n비율 {ratio*100:.1f}%"
        lines.append(f'{node_id} [label="{node_label}", style=filled, fillcolor="{fill_color}", shape=box];')
    else:
        feat_idx = tree.feature[node_id]
        feat = feature_names[feat_idx]
        feat_kor = kor_name[feat]
        threshold = tree.threshold[node_id]
        node_label = f"{feat_kor} <= {threshold:.2f} ?\\n인원 {n_total}명 (뇌졸중 {n_yes}명)\\n비율 {ratio*100:.1f}%"
        lines.append(f'{node_id} [label="{node_label}", style=filled, fillcolor="#fff4cc", shape=box];')

        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]

        lines.append(f'{node_id} -> {left_id} [label="예"];')
        lines.append(f'{node_id} -> {right_id} [label="아니오"];')

        lines.extend(build_dot(left_id))
        lines.extend(build_dot(right_id))

    return lines

dot_lines = build_dot(0)
dot_string = "digraph Tree {\nnode [fontname=\"Malgun Gothic\"];\nedge [fontname=\"Malgun Gothic\"];\n" + "\n".join(dot_lines) + "\n}"

st.graphviz_chart(dot_string)

# 리프 노드 통계
leaf_ids = [i for i in range(tree.node_count)
            if tree.children_left[i] == -1 and tree.children_right[i] == -1]
leaf_count = len(leaf_ids)
no_answer_count = 0
for leaf_id in leaf_ids:
    _, n_yes, ratio = get_node_info(leaf_id)
    if ratio < 0.5:
        no_answer_count += 1

used_feature_indices = set(f for f in tree.feature if f != -2)
used_features = [feature_names[i] for i in used_feature_indices]
used_features_kor = [kor_name[f] for f in used_features]
not_used_features_kor = [kor_name[f] for f in selected_features if f not in used_features]

st.markdown(f"""
<div class="info-box">
🔎 <b>답을 내는 마디(리프 노드) 개수</b>: 총 {leaf_count}칸<br>
🔎 <b>그 중 "정상(아님)"이라고 답하는 마디</b>: {no_answer_count}칸<br>
🔎 <b>이 나무가 실제로 사용한 속성</b>: {", ".join(used_features_kor) if used_features_kor else "없음"}<br>
🔎 <b>선택했지만 나무가 사용하지 않은 속성</b>: {", ".join(not_used_features_kor) if not_used_features_kor else "없음 (모두 사용함)"}
</div>
""", unsafe_allow_html=True)

with st.expander("📖 해석 보기 (트리 구조는 어떻게 읽나요?)"):
    st.write(f"""
    - 트리는 맨 위 뿌리 마디에서 시작해서, 질문에 "예" 또는 "아니오"로 답하며
      아래로 내려갑니다. 이 트리는 최대 **3번**까지만 질문할 수 있도록
      제한했습니다.
    - 각 마디에는 그 자리까지 내려온 **훈련용 사람 수**, 그중 **실제로
      뇌졸중이었던 사람 수**, 그리고 그 **비율**이 적혀 있습니다.
    - 네모 칸 중 더 이상 질문하지 않고 답을 내리는 마디를 **리프 노드**라고
      부릅니다. 파란색 리프는 "정상(뇌졸중 아님)"이라고 답하는 마디이고,
      빨간색 리프는 "뇌졸중"이라고 답하는 마디입니다.
    - 이 트리는 한 마디에 도착한 사람이 **5명 미만**이면 더 이상 나누지
      않도록 설정했습니다. 사람 수가 너무 적으면 그 안에서의 비율이
      우연에 의해 크게 흔들릴 수 있기 때문입니다.
    - 리프 노드가 총 **{leaf_count}개**이고, 그중 **{no_answer_count}개**가
      "정상"이라고 답합니다. 이는 대부분의 사람들이 뇌졸중이 아니기 때문에,
      트리가 "정상"이라고 답하는 경우가 더 많이 생기는 것입니다.
    - 트리가 선택한 속성 중 일부만 실제로 사용했을 수도 있습니다. 이는
      트리가 질문을 3번까지만 할 수 있기 때문에, 가장 정보가 되는 속성부터
      우선적으로 사용하고 나머지는 미처 사용하지 못했기 때문입니다.
    """)
