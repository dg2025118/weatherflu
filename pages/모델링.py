
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    recall_score
)

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(
    page_title="뉴욕시 날씨와 독감 예측",
    layout="wide"
)

st.title("🌡️ 뉴욕시 날씨와 독감 발생 예측")
st.write(
    "날씨 데이터를 활용하여 주간 독감 환자 수를 예측하고 "
    "세 가지 머신러닝 모델의 성능을 비교합니다."
)

# -------------------------
# 2. 데이터 업로드
# -------------------------
uploaded_file = st.file_uploader(
    "분석할 CSV 파일을 업로드하세요.",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("1. 데이터 확인")
    st.write("데이터 크기:", df.shape)
    st.dataframe(df.head())

    # -------------------------
    # 3. 사용할 변수 선택
    # -------------------------
    target = "뉴욕시합계"

    features = [
        "TMIN_AVG",
        "TMAX_AVG",
        "TAVG_AVG",
        "PRCP_AVG",
        "SNOW_AVG",
        "SNWD_AVG"
    ]

    required_columns = features + [target]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"CSV 파일에 필요한 열이 없습니다: {missing_columns}"
        )
        st.stop()

    # -------------------------
    # 4. 데이터 전처리
    # -------------------------
    data = df.copy()

    # 날짜가 있다면 시간 순서대로 정렬
    if "주마지막날짜" in data.columns:
        data["주마지막날짜"] = pd.to_datetime(
            data["주마지막날짜"],
            errors="coerce"
        )
        data = data.sort_values("주마지막날짜")

    # 숫자가 아닌 값은 결측값으로 처리
    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    # 결측값이 있는 행 제거
    data = data.dropna(subset=required_columns)

    if len(data) < 10:
        st.error("분석에 사용할 데이터가 너무 적습니다.")
        st.stop()

    X = data[features]
    y = data[target]

    st.write("전처리 후 데이터:", len(data), "행")
    st.write("결측값 제거 후 분석을 진행합니다.")

    # -------------------------
    # 5. 학습 데이터와 테스트 데이터 분리
    # -------------------------
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # -------------------------
    # 6. 모델 생성
    # -------------------------
    models = {
        "선형회귀": LinearRegression(),

        "결정트리회귀": DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        ),

        "랜덤포레스트회귀": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        )
    }

    # -------------------------
    # 7. 기준 모델
    # -------------------------
    baseline_predictions = np.full(
        len(y_test),
        y_train.mean()
    )

    # 고위험 주간 기준: 학습 데이터의 상위 25% 기준
    threshold = y_train.quantile(0.75)

    results = []
    predictions = {}

    # -------------------------
    # 8. 모델 학습 및 평가
    # -------------------------
    def evaluate_model(name, actual, predicted):

        # 독감 환자 수는 음수가 될 수 없으므로 0으로 제한
        predicted = np.maximum(predicted, 0)

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        r2 = r2_score(actual, predicted)

        # 실제 고위험 주간과 예측 고위험 주간 구분
        actual_high = (actual >= threshold).astype(int)
        predicted_high = (predicted >= threshold).astype(int)

        accuracy = accuracy_score(
            actual_high,
            predicted_high
        )

        recall = recall_score(
            actual_high,
            predicted_high,
            zero_division=0
        )

        return {
            "모델": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "정확도": accuracy,
            "재현율": recall
        }, predicted

    # 기준 모델 평가
    baseline_result, baseline_pred = evaluate_model(
        "기준 모델",
        y_test,
        baseline_predictions
    )

    results.append(baseline_result)
    predictions["기준 모델"] = baseline_pred

    # 나머지 세 모델 평가
    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        result, pred = evaluate_model(
            name,
            y_test,
            pred
        )

        results.append(result)
        predictions[name] = pred

    # -------------------------
    # 9. 평가 결과 출력
    # -------------------------
    st.subheader("2. 모델별 평가지표")

    result_df = pd.DataFrame(results)

    display_df = result_df.copy()
    display_df["정확도"] *= 100
    display_df["재현율"] *= 100

    st.dataframe(
        display_df.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R²": "{:.3f}",
            "정확도": "{:.2f}%",
            "재현율": "{:.2f}%"
        }),
        use_container_width=True
    )

    st.caption(
        f"고위험 주간 기준: 독감 환자 수 {threshold:.0f}명 이상 "
        "(학습 데이터의 상위 25% 기준)"
    )

    # -------------------------
    # 10. 실제값과 예측값 비교
    # -------------------------
    st.subheader("3. 실제 독감 환자 수와 예측값 비교")

    chart_data = pd.DataFrame({
        "실제 독감 환자 수": y_test.values,
        "기준 모델": predictions["기준 모델"],
        "선형회귀": predictions["선형회귀"],
        "결정트리회귀": predictions["결정트리회귀"],
        "랜덤포레스트회귀": predictions["랜덤포레스트회귀"]
    })

    if "주마지막날짜" in data.columns:
        chart_data.index = data["주마지막날짜"].iloc[split_index:]

    st.line_chart(chart_data)

    # -------------------------
    # 11. 모델별 중요 변수
    # -------------------------
    st.subheader("4. 독감 예측에 영향을 준 날씨 변수")

    tree_model = models["결정트리회귀"]
    forest_model = models["랜덤포레스트회귀"]

    importance_df = pd.DataFrame({
        "날씨 변수": features,
        "결정트리 중요도": tree_model.feature_importances_,
        "랜덤포레스트 중요도": forest_model.feature_importances_
    }).set_index("날씨 변수")

    st.bar_chart(importance_df)

    # -------------------------
    # 12. 최종 해석
    # -------------------------
    st.subheader("5. 분석 결과 해석")

    best_recall = result_df.loc[
        result_df["모델"] != "기준 모델"
    ].sort_values("재현율", ascending=False).iloc[0]

    st.write(
        f"재현율이 가장 높은 모델은 **{best_recall['모델']}**입니다."
    )

    st.write(
        f"재현율은 {best_recall['재현율'] * 100:.2f}%입니다."
    )

    st.write(
        "날씨만으로 독감 발생을 정확하게 예측하기는 어려울 수 있습니다. "
        "예방접종, 바이러스 유행, 사람들의 생활 습관 등 "
        "다른 요인도 영향을 줄 수 있기 때문입니다."
    )

else:
    st.info("위쪽에서 분석할 CSV 파일을 업로드해 주세요.")

