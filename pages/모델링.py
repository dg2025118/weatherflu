
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 화면 설정
st.set_page_config(
    page_title="뉴욕시 날씨와 독감 예측",
    layout="wide"
)

st.title("뉴욕시 날씨와 독감 발생량 예측")
st.write("뉴욕시 날씨 데이터를 이용한 독감 발생량 회귀 분석")

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
df["주차_sin"] = np.sin(
    2 * np.pi * df["CDC주차"] / 53
)

df["주차_cos"] = np.cos(
    2 * np.pi * df["CDC주차"] / 53
)

features += ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

# 5. 전체 데이터로 모델 학습
@st.cache_data
def train_models(X, y):

    # 선형 회귀
    linear = LinearRegression()
    linear.fit(X, y)
    linear_pred = linear.predict(X)

    # 결정 트리 회귀
    tree = DecisionTreeRegressor(
        max_depth=None,
        min_samples_leaf=1,
        random_state=42
    )
    tree.fit(X, y)
    tree_pred = tree.predict(X)

    # 랜덤 포레스트 회귀
    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1
    )
    forest.fit(X, y)
    forest_pred = forest.predict(X)

    return linear_pred, tree_pred, forest_pred

with st.spinner("모델을 학습하고 있습니다..."):
    linear_pred, tree_pred, forest_pred = train_models(X, y)

# 6. 베이스라인
baseline_pred = np.full(len(y), y.mean())

# 7. 평가지표 계산
def evaluate(name, actual, predicted):
    return {
        "모델": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R²": r2_score(actual, predicted)
    }

results = [
    evaluate("베이스라인", y, baseline_pred),
    evaluate("선형 회귀", y, linear_pred),
    evaluate("결정 트리 회귀", y, tree_pred),
    evaluate("랜덤 포레스트 회귀", y, forest_pred)
]

result_df = pd.DataFrame(results)

# 8. 결과 출력
st.subheader("모델별 평가지표")

st.dataframe(
    result_df.round(3),
    use_container_width=True,
    hide_index=True
)

# 9. MAE 비교
st.subheader("MAE 결과 비교")

tree_mae = result_df.loc[
    result_df["모델"] == "결정 트리 회귀", "MAE"
].iloc[0]

other_mae = result_df[
    result_df["모델"].isin(
        ["선형 회귀", "랜덤 포레스트 회귀"]
    )
]["MAE"].min()

if tree_mae < other_mae:
    st.success("결정 트리 회귀의 MAE가 세 회귀 모델 중 가장 작습니다.")
else:
    st.info("결정 트리 회귀의 MAE가 가장 작지는 않습니다.")

# 10. 그래프
st.subheader("실제 독감 발생량과 예측값 비교")

chart_df = pd.DataFrame({
    "실제 독감 발생량": y.values,
    "베이스라인": baseline_pred,
    "선형 회귀": linear_pred,
    "결정 트리 회귀": tree_pred,
    "랜덤 포레스트 회귀": forest_pred
}, index=df["주마지막날짜"])

st.line_chart(chart_df, height=450)

# 11. 데이터 정보
st.subheader("데이터 정보")

col1, col2 = st.columns(2)

col1.metric("전체 데이터", f"{len(df)}행")
col2.metric("입력 변수", f"{len(features)}개")

# 12. 하이퍼파라미터
st.subheader("모델 설정값")

st.write("**결정 트리 회귀**")
st.json({
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": 42
})

st.write("**랜덤 포레스트 회귀**")
st.json({
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": 1
})

st.warning(
    "주의: 전체 데이터를 학습하고 같은 데이터로 평가했습니다. "
    "따라서 새로운 데이터에 대한 예측 성능이 아닌 "
    "학습 데이터에 대한 적합도를 나타냅니다."
)
