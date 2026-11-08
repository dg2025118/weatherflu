
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="뉴욕 날씨와 독감 예측", layout="wide")

st.title("뉴욕시 날씨와 독감 발생량 예측")
st.write("날씨 데이터를 이용해 주간 독감 발생량을 예측하고 3가지 모델을 비교합니다.")

# 1. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%95%EB%A6%AC.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.subheader("1. 데이터 확인")
st.write(f"전체 데이터: **{len(df)}행**")
st.dataframe(df.head())

# 2. 사용할 변수 선택
target = "뉴욕시합계"

weather_features = [
    "TMIN_AVG",
    "TMAX_AVG",
    "TAVG_AVG",
    "PRCP_AVG",
    "SNOW_AVG",
    "SNWD_AVG"
]

required_columns = weather_features + [target, "CDC주차", "주마지막날짜"]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"데이터에 다음 열이 없습니다: {missing_columns}")
    st.stop()

# 3. 데이터 전처리
df["주마지막날짜"] = pd.to_datetime(df["주마지막날짜"], errors="coerce")

for col in weather_features + [target, "CDC주차"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns)
df = df.sort_values("주마지막날짜").reset_index(drop=True)

# CDC 주차를 계절성을 나타내는 변수로 변환
df["주차_sin"] = np.sin(2 * np.pi * df["CDC주차"] / 53)
df["주차_cos"] = np.cos(2 * np.pi * df["CDC주차"] / 53)

features = weather_features + ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

if len(df) < 10:
    st.error("분석할 데이터가 너무 적습니다.")
    st.stop()

# 4. 시간 순서대로 학습 데이터와 테스트 데이터 분리
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

st.subheader("2. 학습 및 테스트 데이터")
st.write(f"학습 데이터: **{len(X_train)}개** (앞의 80%)")
st.write(f"테스트 데이터: **{len(X_test)}개** (뒤의 20%)")

# 5. 베이스라인 설정
# 학습 데이터의 독감 발생량 평균을 모든 주의 예측값으로 사용
baseline_pred = np.full(len(y_test), y_train.mean())

# 6. 비교할 모델 3개 설정
models = {
    "선형 회귀": LinearRegression(),

    "결정 트리 회귀": DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    ),

    "랜덤 포레스트 회귀": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    )
}

# 7. 모델 학습 및 평가
results = []

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "모델": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4)
    }

results.append(
    evaluate_model("베이스라인", y_test, baseline_pred)
)

predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predictions[name] = pred

    results.append(
        evaluate_model(name, y_test, pred)
    )

results_df = pd.DataFrame(results)

# 8. 결과 비교
st.subheader("3. 모델별 평가 결과")
st.dataframe(results_df, use_container_width=True)

st.caption("MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋습니다.")

# MAE가 가장 낮은 모델 선택
model_results = results_df[results_df["모델"] != "베이스라인"]
best_model_name = model_results.loc[
    model_results["MAE"].idxmin(), "모델"
]

st.subheader("4. 가장 낮은 MAE를 기록한 모델")
st.success(f"현재 비교 결과: **{best_model_name}**")

st.write(
    "MAE가 가장 낮은 모델은 평균적인 예측 오차가 가장 작다는 뜻입니다. "
    "다른 평가 지표도 함께 확인해야 합니다."
)

# 9. 모델별 예측 그래프
st.subheader("5. 실제 독감 발생량과 예측값 비교")

chart_data = pd.DataFrame({
    "실제 독감 발생량": y_test.values,
    "베이스라인": baseline_pred,
    "선형 회귀": predictions["선형 회귀"],
    "결정 트리 회귀": predictions["결정 트리 회귀"],
    "랜덤 포레스트 회귀": predictions["랜덤 포레스트 회귀"]
}, index=df["주마지막날짜"].iloc[split_index:])

st.line_chart(chart_data)

# 10. 모델별 주요 설정값
st.subheader("6. 모델의 주요 하이퍼파라미터")

hyperparameters = pd.DataFrame({
    "모델": ["선형 회귀", "결정 트리 회귀", "랜덤 포레스트 회귀"],
    "주요 설정": [
        "기본 설정 사용",
        "max_depth=5, min_samples_leaf=5",
        "n_estimators=300, max_depth=8, min_samples_leaf=3"
    ]
})

st.table(hyperparameters)

# 11. 결과 해석
st.subheader("7. 결과 해석")

st.write("""
- **베이스라인**: 학습 데이터의 평균 독감 발생량으로 예측합니다.
- **선형 회귀**: 날씨와 독감 발생량 사이의 직선적인 관계를 학습합니다.
- **결정 트리 회귀**: 조건을 나누어 독감 발생량을 예측합니다.
- **랜덤 포레스트 회귀**: 여러 결정 트리의 예측을 모아 결과를 만듭니다.

베이스라인보다 성능이 좋은지 확인하면 모델이 예측에 실제로 도움이 되는지 판단할 수 있습니다.
""")

st.info(
    "주의: 이 분석은 날씨와 독감 발생량의 관계를 살펴보는 것이며, "
    "날씨가 독감을 직접 발생시킨다는 뜻은 아닙니다."
)
