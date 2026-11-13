
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

# 4. CDC 주차를 계절 정보로 변환
df["주차_sin"] = np.sin(2 * np.pi * df["CDC주차"] / 53)
df["주차_cos"] = np.cos(2 * np.pi * df["CDC주차"] / 53)

features += ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

# 5. 시간 순서대로 80:20 분리
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# 학습 데이터 안에서만 교차 검증
tscv = TimeSeriesSplit(n_splits=4)

# 6. 선형 회귀
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 7. 결정 트리 회귀 하이퍼파라미터 탐색
tree_params = {
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "min_samples_leaf": [2, 3, 5, 8, 10],
    "min_samples_split": [2, 5, 10]
}

tree_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    tree_params,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1
)

tree_search.fit(X_train, y_train)
tree_model = tree_search.best_estimator_

# 8. 랜덤 포레스트 회귀 하이퍼파라미터 탐색
forest_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [2, 3, 5]
}

forest_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    forest_params,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    n_jobs=-1
)

forest_search.fit(X_train, y_train)
forest_model = forest_search.best_estimator_

# 9. 테스트 데이터 예측
predictions = {
    "선형 회귀": linear_model.predict(X_test),
    "결정 트리 회귀": tree_model.predict(X_test),
    "랜덤 포레스트 회귀": forest_model.predict(X_test)
}

# 10. 베이스라인 예측
baseline_prediction = np.full(
    len(y_test), y_train.mean()
)

predictions["베이스라인"] = baseline_prediction

# 11. 평가지표 계산
results = []

for name, pred in predictions.items():
    results.append({
        "모델": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R²": r2_score(y_test, pred)
    })

result_df = pd.DataFrame(results)

st.subheader("모델별 평가지표 비교")
st.dataframe(
    result_df.round(3),
    use_container_width=True,
    hide_index=True
)

# 12. 결정 트리 MAE 결과 확인
tree_mae = result_df.loc[
    result_df["모델"] == "결정 트리 회귀", "MAE"
].iloc[0]

other_maes = result_df[
    result_df["모델"].isin(["선형 회귀", "랜덤 포레스트 회귀"])
]["MAE"]

st.subheader("결정 트리 회귀 결과")

if tree_mae < other_maes.min():
    st.success("결정 트리 회귀의 MAE가 세 회귀 모델 중 가장 작습니다.")
else:
    st.info(
        "현재 테스트 데이터에서는 결정 트리 회귀의 MAE가 "
        "가장 작지 않습니다. 실제 결과를 그대로 확인하세요."
    )

# 13. 최적 하이퍼파라미터
st.subheader("모델별 최적 설정")

st.write("결정 트리 회귀")
st.json(tree_search.best_params_)

st.write("랜덤 포레스트 회귀")
st.json(forest_search.best_params_)

# 14. 실제값과 예측값 비교
st.subheader("실제 독감 발생량과 예측값 비교")

chart_df = pd.DataFrame({
    "실제 독감 발생량": y_test.values,
    "선형 회귀": predictions["선형 회귀"],
    "결정 트리 회귀": predictions["결정 트리 회귀"],
    "랜덤 포레스트 회귀": predictions["랜덤 포레스트 회귀"],
    "베이스라인": predictions["베이스라인"]
}, index=df["주마지막날짜"].iloc[split:])

st.line_chart(chart_df, height=500)

# 15. 데이터 개수
st.subheader("데이터 분리 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}행")
col2.metric("학습 데이터", f"{len(X_train)}행")
col3.metric("테스트 데이터", f"{len(X_test)}행")

st.caption(
    "시간 순서대로 앞의 80%를 학습에 사용하고 "
    "뒤의 20%를 최종 평가에 사용했습니다."
)
