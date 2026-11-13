
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="뉴욕 날씨와 독감 분석", layout="wide")

st.title("뉴욕 날씨와 독감 발생량 분석")
st.write("선형 회귀, 릿지 회귀, 랜덤 포레스트 회귀 비교")

# 1. 데이터 불러오기
DATA_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%95%EB%A6%AC.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# 2. 필요한 열 선택
features = [
    "TMIN_AVG",
    "TMAX_AVG",
    "TAVG_AVG",
    "PRCP_AVG",
    "SNOW_AVG",
    "SNWD_AVG"
]

target = "뉴욕시합계"

required = features + [target, "주마지막날짜", "CDC주차"]

if not all(col in df.columns for col in required):
    st.error("CSV 파일의 열 이름을 확인해주세요.")
    st.stop()

df["주마지막날짜"] = pd.to_datetime(
    df["주마지막날짜"], errors="coerce"
)

for col in features + [target, "CDC주차"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required)
df = df.sort_values("주마지막날짜").reset_index(drop=True)

# 3. 주차 계절성 반영
df["주차_sin"] = np.sin(2 * np.pi * df["CDC주차"] / 52)
df["주차_cos"] = np.cos(2 * np.pi * df["CDC주차"] / 52)

features += ["주차_sin", "주차_cos"]

X = df[features]
y = df[target]

if len(df) < 30:
    st.error("분석에 사용할 데이터가 너무 적습니다.")
    st.stop()

# 4. 시간 순서대로 데이터 분리
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_train_val = X.iloc[:val_end]
y_train_val = y.iloc[:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

# 5. 랜덤 포레스트 설정 비교
st.subheader("랜덤 포레스트 설정 비교")

configs = [
    {"n_estimators": 80, "max_depth": 5, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
    {"n_estimators": 120, "max_depth": 8, "min_samples_leaf": 3},
    {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2},
    {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 3}
]

best_config = None
best_val_mae = float("inf")

for config in configs:
    model = RandomForestRegressor(
        **config,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, pred)

    if mae < best_val_mae:
        best_val_mae = mae
        best_config = config

st.write("선택된 랜덤 포레스트 설정")
st.json(best_config)
st.write(f"검증 데이터 MAE: {best_val_mae:,.2f}")

# 6. 세 가지 회귀 모델 학습
models = {
    "선형 회귀": LinearRegression(),

    "릿지 회귀": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),

    "랜덤 포레스트 회귀": RandomForestRegressor(
        **best_config,
        random_state=42,
        n_jobs=1
    )
}

predictions = {}

for name, model in models.items():
    model.fit(X_train_val, y_train_val)
    predictions[name] = model.predict(X_test)

# 7. 베이스라인
baseline_value = y_train_val.mean()

predictions["베이스라인"] = np.full(
    len(y_test), baseline_value
)

# 8. 평가지표 계산
results = []

for name, pred in predictions.items():
    results.append({
        "모델": name,
        "MAE": round(
            mean_absolute_error(y_test, pred), 3
        ),
        "RMSE": round(
            np.sqrt(mean_squared_error(y_test, pred)), 3
        ),
        "R²": round(
            r2_score(y_test, pred), 3
        )
    })

result_df = pd.DataFrame(results)

# 9. 결과 출력
st.subheader("모델별 평가 지표")
st.caption("모든 모델은 동일한 최종 평가 데이터로 비교합니다.")

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)

# 10. MAE 비교
st.subheader("MAE 기준 모델 비교")

best_model = result_df.loc[result_df["MAE"].idxmin()]

st.success(
    f"현재 가장 낮은 MAE 모델: {best_model['모델']} "
    f"(MAE: {best_model['MAE']:,.3f})"
)

rf_mae = result_df.loc[
    result_df["모델"] == "랜덤 포레스트 회귀", "MAE"
].iloc[0]

other_mae = result_df[
    result_df["모델"] != "랜덤 포레스트 회귀"
]["MAE"].min()

if rf_mae < other_mae:
    st.success(
        "랜덤 포레스트 회귀가 비교 모델 중 MAE가 가장 낮습니다."
    )
else:
    st.info(
        "이번 평가에서는 다른 모델의 MAE가 더 낮았습니다."
    )

# 11. 실제값과 예측값 그래프
st.subheader("실제 독감 발생량과 예측값 비교")

chart_df = pd.DataFrame({
    "실제 독감 발생량": y_test.values,
    "랜덤 포레스트 예측": predictions["랜덤 포레스트 회귀"],
    "선형 회귀 예측": predictions["선형 회귀"],
    "릿지 회귀 예측": predictions["릿지 회귀"]
}, index=df["주마지막날짜"].iloc[val_end:])

st.line_chart(chart_df)

# 12. 데이터 정보
st.subheader("분석 정보")

st.write(f"- 전체 데이터: {len(df)}행")
st.write(f"- 학습 및 검증 데이터: {val_end}행")
st.write(f"- 최종 평가 데이터: {len(y_test)}행")
st.write(f"- 입력 변수: {', '.join(features)}")
st.write(f"- 랜덤 포레스트 설정: {best_config}")
st.write("- 릿지 회귀 alpha: 1.0")

st.warning(
    "MAE는 실제값과 예측값의 평균적인 차이입니다. "
    "낮을수록 예측 성능이 좋습니다."
)
