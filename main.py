import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_weekly_averages_2009-10-03_to_2022-12-29.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="기온과 독감", layout="wide")
st.title("🌡️ 기온이 낮아질수록 독감 의심 환자가 많아질까?")


@st.cache_data
def load(url):
    return pd.read_csv(url)


weather_raw = load(WEATHER_URL)
flu_raw = load(FLU_URL)

# ---------- 사이드바: 열 직접 고르기 ----------
st.sidebar.header("⚙️ 설정")
w_date = st.sidebar.selectbox("날씨 날짜 열", weather_raw.columns)
temp_col = st.sidebar.selectbox("기온 열", weather_raw.columns)
f_date = st.sidebar.selectbox("독감 날짜 열", flu_raw.columns)
flu_col = st.sidebar.selectbox("독감 환자 열", flu_raw.columns)
remove_out = st.sidebar.radio("이상치 처리", ["유지", "제거"])

# ---------- 1. 데이터 개요 ----------
st.header("1️⃣ 데이터 개요")
for name, df in [("날씨", weather_raw), ("인플루엔자", flu_raw)]:
    st.subheader(name)
    st.write(f"행 {len(df)}개 / 열 {df.shape[1]}개")
    st.dataframe(pd.DataFrame({
        "자료형": df.dtypes.astype(str),
        "빈 값 개수": df.isna().sum(),
    }))

# ---------- 2. 전처리 ----------
w = pd.DataFrame({"DATE": pd.to_datetime(weather_raw[w_date], errors="coerce"),
                  "TEMP": pd.to_numeric(weather_raw[temp_col], errors="coerce")})
f = pd.DataFrame({"DATE": pd.to_datetime(flu_raw[f_date], errors="coerce"),
                  "FLU": pd.to_numeric(flu_raw[flu_col], errors="coerce")})

merged = pd.merge(f, w, on="DATE", how="inner")
n_merged = len(merged)

clean = merged.dropna().sort_values("DATE")        # 결측치 삭제
n_clean = len(clean)


def is_outlier(s):  # IQR 방법
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


out = is_outlier(clean["TEMP"]) | is_outlier(clean["FLU"])
data = clean[~out] if remove_out == "제거" else clean
n_final = len(data)

st.header("2️⃣ 결측치·이상치 처리")
st.write("- 결측치: 값이 빈 주는 **삭제**")
st.write(f"- 이상치: IQR 방법으로 찾았고, 현재 설정은 **{remove_out}**")
st.table(pd.DataFrame({
    "단계": ["두 데이터 합친 후", "결측치 삭제 후", "이상치 처리 후"],
    "행 수": [n_merged, n_clean, n_final],
}))
st.write(f"결측치 {n_merged - n_clean}개 삭제, 이상치 {int(out.sum())}개 발견")

if n_final < 5:
    st.error("데이터가 너무 적어요. 사이드바에서 열 선택을 확인하세요.")
    st.stop()

# ---------- 3. 그래프 ----------
st.header("3️⃣ 질문: 기온이 낮을수록 환자가 많을까?")
r = data["TEMP"].corr(data["FLU"])

# 그래프 1
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["DATE"], y=data["FLU"], name="환자 수",
                         line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=data["DATE"], y=data["TEMP"], name="기온(위쪽이 추움)",
                         yaxis="y2", line=dict(color="royalblue")))
fig.update_layout(
    title="주간 환자 수와 기온의 변화",
    xaxis_title="날짜", yaxis_title="독감 환자 수",
    yaxis2=dict(title="기온 (뒤집힘)", overlaying="y", side="right", autorange="reversed"),
)
st.plotly_chart(fig)
top = data[data["FLU"] >= data["FLU"].quantile(0.9)]
st.info(f"🔎 찾아낸 것: 환자가 가장 많은 주들은 기온이 낮은 시기에 몰려 있는지 확인해요.\n\n"
        f"📊 근거: 환자 상위 10% 주의 평균 기온 {top['TEMP'].mean():.1f} "
        f"vs 전체 평균 {data['TEMP'].mean():.1f}")

# 그래프 2
fig = px.scatter(data, x="TEMP", y="FLU", trendline="ols",
                 title="기온과 환자 수의 관계 (점 하나 = 1주)",
                 labels={"TEMP": "주간 평균 기온", "FLU": "독감 환자 수"})
st.plotly_chart(fig)
st.info(f"🔎 찾아낸 것: r이 음수이면 '기온이 낮을수록 환자가 많다'는 뜻이에요.\n\n"
        f"📊 근거: 상관계수 r = {r:.3f}, 분석한 주 {len(data)}개")

st.text_area("✍️ 우리 팀의 해석과 결론 (한계점 포함)")
