
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="뉴욕시 날씨와 독감 예측",
    layout="wide"
)

st.title("뉴욕시 날씨와 독감 발생량 예측")

# 1. 데이터 불러오기
URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%95%EB%A6%AC.csv"

@st.cache_data
def load_data():
    return pd.read_csv(URL)

df = load_data()

# 2. 사용할 변수
features = [
    "TMIN_AVG",
    "TMAX_AVG",
    "TAVG_AVG",
    "PRCP_AVG",
    "SNOW_AVG",
    "SNWD_AVG"
]

target = "뉴욕시합계"

# 3. 데이터 정리
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

features += ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

# 5. 전체 데이터 사용
# 학습 데이터와 테스트 데이터를 나누지 않음

# 6. 베이스라인
baseline_prediction = np.full(len(y), y.mean())

# 7. 선형 회귀
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_prediction = linear_model.predict(X)

# 8. 결정 트리 회귀
tree_model = DecisionTreeRegressor(
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

tree_model.fit(X, y)
tree_prediction = tree_model.predict(X)

# 9. 랜덤 포레스트 회귀
forest_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42
)

forest_model.fit(X, y)
forest_prediction = forest_model.predict(X)

# 10. 예측값 정리
predictions = {
    "베이스라인": baseline_prediction,
    "선형 회귀": linear_prediction,
    "결정 트리 회귀": tree_prediction,
    "랜덤 포레스트 회귀": forest_prediction
}

# 11. 평가지표 계산
results = []

for name, pred in predictions.items():
    results.append({
        "모델": name,
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R²": r2_score(y, pred)
    })

result_df = pd.DataFrame(results)

# 12. 결과 출력
st.subheader("모델별 평가지표 비교")

st.dataframe(
    result_df.round(3),
    use_container_width=True,
    hide_index=True
)

# 13. MAE 비교
st.subheader("MAE 결과 비교")

tree_mae = result_df.loc[
    result_df["모델"] == "결정 트리 회귀", "MAE"
].iloc[0]

other_maes = result_df[
    result_df["모델"].isin(["선형 회귀", "랜덤 포레스트 회귀"])
]["MAE"]

if tree_mae < other_maes.min():
    st.success("결정 트리 회귀의 MAE가 가장 작습니다.")
else:
    st.info("결정 트리 회귀의 MAE가 가장 작지는 않습니다.")

# 14. 실제값과 예측값 그래프
st.subheader("실제 독감 발생량과 예측값 비교")

chart_df = pd.DataFrame({
    "실제 독감 발생량": y.values,
    "베이스라인": baseline_prediction,
    "선형 회귀": linear_prediction,
    "결정 트리 회귀": tree_prediction,
    "랜덤 포레스트 회귀": forest_prediction
}, index=df["주마지막날짜"])

st.line_chart(chart_df, height=500)

# 15. 데이터 정보
st.subheader("사용 데이터 정보")

col1, col2 = st.columns(2)

col1.metric("전체 데이터 행 수", f"{len(df)}행")
col2.metric("입력 변수 개수", f"{len(features)}개")

st.write("**결정 트리 회귀 설정값**")
st.write({
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 42
})

st.write("**랜덤 포레스트 회귀 설정값**")
st.write({
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 42
})

st.warning(
    "주의: 전체 데이터를 학습한 뒤 같은 데이터로 평가했습니다. "
    "따라서 이 평가지표는 새로운 데이터에 대한 예측 성능이 아니라 "
    "학습 데이터에 대한 적합도를 나타냅니다."
)
