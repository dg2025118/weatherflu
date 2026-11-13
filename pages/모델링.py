
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="뉴욕시 날씨와 독감 발생량 예측",
    layout="wide"
)

st.title("뉴욕시 날씨와 독감 발생량 예측")
st.write("전체 데이터를 학습하여 실제 독감 발생량과 예측값을 비교합니다.")

# 1. 데이터 불러오기
URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%95%EB%A6%AC.csv"

@st.cache_data
def load_data():
    return pd.read_csv(URL)

df = load_data()

# 2. 사용할 데이터 변수
features = [
    "TMIN_AVG",
    "TMAX_AVG",
    "TAVG_AVG",
    "PRCP_AVG",
    "SNOW_AVG",
    "SNWD_AVG"
]

target = "뉴욕시합계"

# 3. 날짜와 숫자 데이터 정리
df["주마지막날짜"] = pd.to_datetime(
    df["주마지막날짜"], errors="coerce"
)

df["CDC주차"] = pd.to_numeric(
    df["CDC주차"], errors="coerce"
)

for col in features + [target]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=features + [target, "CDC주차", "주마지막날짜"]
)

df = df.sort_values("주마지막날짜").reset_index(drop=True)

# 4. 계절 정보 추가
df["주차_sin"] = np.sin(2 * np.pi * df["CDC주차"] / 53)
df["주차_cos"] = np.cos(2 * np.pi * df["CDC주차"] / 53)

features = features + ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

# 5. 전체 데이터를 이용하여 모델 학습
linear_model = LinearRegression()

tree_model = DecisionTreeRegressor(
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

forest_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

linear_model.fit(X, y)
tree_model.fit(X, y)
forest_model.fit(X, y)

# 6. 전체 데이터 예측
df["선형 회귀"] = np.maximum(linear_model.predict(X), 0)
df["결정 트리 회귀"] = np.maximum(tree_model.predict(X), 0)
df["랜덤 포레스트 회귀"] = np.maximum(forest_model.predict(X), 0)

# 베이스라인: 전체 데이터 평균
df["베이스라인"] = y.mean()

# 7. 그래프 만들기
st.subheader("실제 독감 발생량과 예측값 비교")

chart_df = df.set_index("주마지막날짜")[
    [
        target,
        "베이스라인",
        "선형 회귀",
        "결정 트리 회귀",
        "랜덤 포레스트 회귀"
    ]
]

chart_df = chart_df.rename(
    columns={target: "실제 독감 발생량"}
)

st.line_chart(chart_df, height=500)

# 8. 예측 결과 확인
st.subheader("예측 결과 데이터")

st.dataframe(
    df[
        [
            "주마지막날짜",
            target,
            "베이스라인",
            "선형 회귀",
            "결정 트리 회귀",
            "랜덤 포레스트 회귀"
        ]
    ],
    use_container_width=True
)

st.success("전체 데이터 학습 및 예측이 완료되었습니다.")

st.warning(
    "주의: 전체 데이터를 학습한 뒤 같은 데이터로 예측한 결과입니다. "
    "실제 새로운 데이터에 대한 예측 성능을 나타내지는 않습니다."
)
