
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =================================
# 1. 페이지 설정
# =================================

st.set_page_config(
    page_title="뉴욕시 날씨와 독감 분석",
    layout="wide"
)

st.title("🌡️ 뉴욕시 날씨와 독감 환자 수 예측")
st.write(
    "날씨와 계절 정보를 이용하여 "
    "주간 독감 환자 수를 예측하는 선형회귀 모델입니다."
)

# =================================
# 2. GitHub 데이터 불러오기
# =================================

data_url = (
    "https://raw.githubusercontent.com/dg2025118/weatherflu/"
    "refs/heads/main/%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%95%EB%A6%AC.csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(data_url, encoding="utf-8-sig")

try:
    df = load_data()
    st.success("GitHub 데이터 불러오기 성공!")

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# =================================
# 3. 데이터 확인
# =================================

st.subheader("1. 데이터 확인")

col1, col2 = st.columns(2)

col1.metric("전체 데이터 행 수", f"{len(df)}개")
col2.metric("전체 열 수", f"{len(df.columns)}개")

st.dataframe(df.head(10), use_container_width=True)

# =================================
# 4. 사용할 변수 설정
# =================================

target = "뉴욕시합계"

features = [
    "TAVG_AVG",
    "PRCP_AVG",
    "SNOW_AVG",
    "SNWD_AVG"
]

required_columns = features + [
    target,
    "CDC주차",
    "주마지막날짜"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(f"필요한 열이 없습니다: {missing}")
    st.stop()

# =================================
# 5. 데이터 전처리
# =================================

data = df.copy()

data["주마지막날짜"] = pd.to_datetime(
    data["주마지막날짜"],
    errors="coerce"
)

for col in features + [target, "CDC주차"]:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

data = data.dropna(
    subset=required_columns
)

data = data.sort_values(
    "주마지막날짜"
).reset_index(drop=True)

# CDC 주차를 계절 정보로 변환
data["주차_sin"] = np.sin(
    2 * np.pi * data["CDC주차"] / 53
)

data["주차_cos"] = np.cos(
    2 * np.pi * data["CDC주차"] / 53
)

features = features + [
    "주차_sin",
    "주차_cos"
]

X = data[features]
y = data[target]

# =================================
# 6. 학습 데이터와 테스트 데이터 분리
# =================================

split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

st.subheader("2. 학습 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

col1.metric("학습 데이터", f"{len(X_train)}개")
col2.metric("테스트 데이터", f"{len(X_test)}개")

# =================================
# 7. 선형회귀 모델 학습
# =================================

model = LinearRegression()

model.fit(X_train, y_train)

# 테스트 데이터 예측
predictions = model.predict(X_test)

# 독감 환자 수는 음수가 될 수 없으므로
predictions = np.maximum(predictions, 0)

# =================================
# 8. 평가지표 계산
# =================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

# 기준 모델: 학습 데이터 평균으로 예측
baseline_predictions = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)

# =================================
# 9. 결과 출력
# =================================

st.subheader("3. 선형회귀 모델 평가지표")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R²", f"{r2:.3f}")

st.write("평가지표 해석")
st.write("- MAE: 실제 환자 수와 예측 환자 수의 평균 차이")
st.write("- RMSE: 큰 예측 오차에 더 큰 영향을 받는 지표")
st.write("- R²: 모델이 실제 데이터의 변화를 얼마나 설명하는지 나타내는 지표")

# 기준 모델 비교
st.subheader("4. 기준 모델과 비교")

comparison = pd.DataFrame({
    "모델": ["기준 모델", "선형회귀"],
    "MAE": [baseline_mae, mae],
    "RMSE": [baseline_rmse, rmse],
    "R²": [baseline_r2, r2]
})

st.dataframe(
    comparison.style.format({
        "MAE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)

# =================================
# 10. 실제값과 예측값 비교 그래프
# =================================

st.subheader("5. 실제 독감 환자 수와 예측값")

chart_data = pd.DataFrame({
    "실제 독감 환자 수": y_test.values,
    "선형회귀 예측값": predictions
})

chart_data.index = (
    data["주마지막날짜"]
    .iloc[split_index:]
    .dt.strftime("%Y-%m-%d")
)

st.line_chart(chart_data)

# =================================
# 11. 예측 결과 확인
# =================================

st.subheader("6. 예측 결과 확인")

result = pd.DataFrame({
    "날짜": data["주마지막날짜"].iloc[split_index:].values,
    "실제 환자 수": y_test.values,
    "예측 환자 수": np.round(predictions, 1)
})

result["예측 오차"] = (
    result["실제 환자 수"] - result["예측 환자 수"]
)

st.dataframe(
    result,
    use_container_width=True
)

# =================================
# 12. 최종 해석
# =================================

st.subheader("7. 분석 결과 해석")

if r2 >= 0:
    st.write(
        f"선형회귀 모델의 R²는 {r2:.3f}입니다. "
        "테스트 데이터에서 일정 수준의 설명력을 보였습니다."
    )
else:
    st.write(
        f"선형회귀 모델의 R²는 {r2:.3f}입니다. "
        "예측 성능이 낮으며, 기준 모델보다 설명력이 부족할 수 있습니다."
    )

if mae < baseline_mae:
    st.success("선형회귀가 기준 모델보다 MAE가 낮습니다.")
else:
    st.warning("선형회귀가 기준 모델보다 MAE가 낮지 않습니다.")

st.write(
    "날씨와 계절 정보만으로 독감 환자 수를 "
    "정확하게 예측하기에는 한계가 있습니다."
)

# =================================
# 13. 결과 다운로드
# =================================

st.subheader("8. 결과 다운로드")

csv = result.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    "예측 결과 CSV 다운로드",
    data=csv,
    file_name="linear_regression_results.csv",
    mime="text/csv"
)
