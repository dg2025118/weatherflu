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

# ---------- 사이드바 ----------
st.sidebar.header("⚙️ 설정")
w_date = st.sidebar.selectbox("날씨 데이터의 날짜 열", weather_raw.columns)
temp_col = st.sidebar.selectbox("기온 열", weather_raw.columns)
f_date = st.sidebar.selectbox("독감 데이터의 날짜 열", flu_raw.columns)
flu_col = st.sidebar.selectbox("독감 환자 열", flu_raw.columns)
convert = st.sidebar.radio("기온 단위", ["화씨(°F) → 섭씨(°C)로 변환", "변환 안 함 (이미 섭씨)"])
remove_out = st.sidebar.radio("이상치 처리", ["유지", "제거"])

# ---------- 1. 데이터 개요 ----------
st.header("1️⃣ 데이터 개요")
c1, c2 = st.columns(2)
for col, name, df in [(c1, "날씨 데이터", weather_raw), (c2, "독감 데이터", flu_raw)]:
    with col:
        st.subheader(name)
        st.write(f"행 {len(df)}개 / 열 {df.shape[1]}개")
        st.dataframe(pd.DataFrame({
            "자료형": df.dtypes.astype(str),
            "빈 값 개수": df.isna().sum(),
        }))

# ---------- 2. 전처리 ----------
temp = pd.to_numeric(weather_raw[temp_col], errors="coerce")
if convert.startswith("화씨"):
    temp = (temp - 32) * 5 / 9

w = pd.DataFrame({"날짜": pd.to_datetime(weather_raw[w_date], errors="coerce"),
                  "기온": temp})
f = pd.DataFrame({"날짜": pd.to_datetime(flu_raw[f_date], errors="coerce"),
                  "환자수": pd.to_numeric(flu_raw[flu_col], errors="coerce")})

merged = pd.merge(f, w, on="날짜", how="inner")
n_merged = len(merged)

clean = merged.dropna().sort_values("날짜")   # 결측치 삭제
n_clean = len(clean)


def is_outlier(s):  # IQR 방법
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


out = is_outlier(clean["기온"]) | is_outlier(clean["환자수"])
data = clean[~out].copy() if remove_out == "제거" else clean.copy()
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

data["월"] = data["날짜"].dt.month

# ---------- 3. 그래프 ----------
st.header("3️⃣ 질문: 기온이 낮을수록 환자가 많을까?")
r = data["기온"].corr(data["환자수"])
st.metric("상관계수 r (음수면 추울수록 환자 많음)", f"{r:.3f}")

# 그래프 1: 시간 흐름
st.subheader("📈 그래프 1. 시간에 따른 환자 수와 기온")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["날짜"], y=data["환자수"], name="독감 환자 수",
                         line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=data["날짜"], y=data["기온"], name="기온 (위쪽이 추움)",
                         yaxis="y2", line=dict(color="royalblue")))
fig.update_layout(
    title="주간 독감 환자 수와 기온의 변화",
    xaxis_title="날짜", yaxis_title="독감 환자 수",
    yaxis2=dict(title="기온 (°C, 뒤집힘)", overlaying="y", side="right",
                autorange="reversed"),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig)
top = data[data["환자수"] >= data["환자수"].quantile(0.9)]
st.info(f"🔎 찾아낸 것: 환자가 가장 많은 주들의 기온이 전체 평균보다 낮은지 비교해요.\n\n"
        f"📊 근거: 환자 상위 10% 주의 평균 기온 {top['기온'].mean():.1f}°C "
        f"vs 전체 평균 {data['기온'].mean():.1f}°C")

# 그래프 2: 산점도
st.subheader("🔵 그래프 2. 기온과 환자 수의 관계")
fig = px.scatter(data, x="기온", y="환자수", color="월",
                 color_continuous_scale="Turbo",
                 title="기온과 독감 환자 수의 관계 (점 하나 = 1주)",
                 labels={"기온": "주간 평균 기온 (°C)", "환자수": "독감 환자 수"})
if data["기온"].std() > 0:
    k, b = np.polyfit(data["기온"], data["환자수"], 1)
    xs = np.linspace(data["기온"].min(), data["기온"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="추세선",
                             line=dict(color="black", dash="dash")))
    slope_txt = f"기온 1°C 상승 시 환자 수 {k:+.2f} 변화"
else:
    slope_txt = "기울기 계산 불가"
st.plotly_chart(fig)
st.info(f"🔎 찾아낸 것: r이 음수이면 '기온이 낮을수록 환자가 많다'는 뜻이에요.\n\n"
        f"📊 근거: 상관계수 r = {r:.3f}, {slope_txt}, 분석한 주 {len(data)}개")

# 그래프 3: 월별 평균
st.subheader("📊 그래프 3. 월별 평균 기온과 평균 환자 수")
m = data.groupby("월")[["기온", "환자수"]].mean().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=m["월"], y=m["환자수"], name="평균 환자 수",
                     marker_color="crimson", opacity=0.7))
fig.add_trace(go.Scatter(x=m["월"], y=m["기온"], name="평균 기온",
                         yaxis="y2", mode="lines+markers",
                         line=dict(color="royalblue")))
fig.update_layout(
    title="월별 평균 독감 환자 수와 평균 기온",
    xaxis=dict(title="월", dtick=1), yaxis_title="평균 독감 환자 수",
    yaxis2=dict(title="평균 기온 (°C)", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig)
peak = m.loc[m["환자수"].idxmax()]
cold = m.loc[m["기온"].idxmin()]
st.info(f"🔎 찾아낸 것: 환자가 가장 많은 달과 가장 추운 달이 같은지 비교해요.\n\n"
        f"📊 근거: 환자 최다 {int(peak['월'])}월({peak['환자수']:,.1f}명), "
        f"최저 기온 {int(cold['월'])}월({cold['기온']:.1f}°C)")

st.text_area("✍️ 우리 팀의 해석과 결론 (한계점 포함)")
