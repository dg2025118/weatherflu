import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_weekly_averages_2009-10-03_to_2022-12-29.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="기온과 독감 의심 환자", layout="wide")
st.title("🌡️ 기온이 낮아질수록 독감 의심 환자가 많아질까?")


# ---------- 유틸 ----------
@st.cache_data
def load(url):
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_date_col(df):
    for key in ["date", "week", "day", "time"]:
        for c in df.columns:
            if key in c.lower():
                if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                    return c
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            if pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8:
                return c
    return df.columns[0]


def prep(df, date_col, prefix):
    d = df.copy()
    d["DATE"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d = d.dropna(subset=["DATE"])
    d["DATE"] = d["DATE"].astype("datetime64[ns]")
    d = d.drop(columns=[date_col], errors="ignore")
    for c in d.columns:
        if c != "DATE" and not pd.api.types.is_numeric_dtype(d[c]):
            conv = pd.to_numeric(
                d[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            if conv.notna().mean() > 0.5:
                d[c] = conv
    num = [c for c in d.columns if c != "DATE" and pd.api.types.is_numeric_dtype(d[c])]
    d = d[["DATE"] + num].groupby("DATE", as_index=False).mean()
    d = d.rename(columns={c: f"{prefix}{c}" for c in num})
    return d.sort_values("DATE").reset_index(drop=True)


def pick_default(options, keywords):
    for k in keywords:
        for i, c in enumerate(options):
            if k in c.lower():
                return i
    return 0


def label(c):
    return c[2:] if c[:2] in ("W_", "F_") else c


# ---------- 데이터 ----------
weather_raw = load(WEATHER_URL)
flu_raw = load(FLU_URL)

st.sidebar.header("⚙️ 설정")
w_date = st.sidebar.selectbox(
    "날씨 데이터의 날짜 열", weather_raw.columns,
    index=list(weather_raw.columns).index(guess_date_col(weather_raw)),
)
f_date = st.sidebar.selectbox(
    "인플루엔자 데이터의 날짜 열", flu_raw.columns,
    index=list(flu_raw.columns).index(guess_date_col(flu_raw)),
)

weather = prep(weather_raw, w_date, "W_")
flu = prep(flu_raw, f_date, "F_")

if weather.empty or flu.empty:
    st.error("날짜 열이 올바르지 않아요. 사이드바에서 다른 열을 선택해보세요.")
    st.write(weather_raw.head())
    st.write(flu_raw.head())
    st.stop()

w_num = [c for c in weather.columns if c != "DATE"]
f_num = [c for c in flu.columns if c != "DATE"]
if not w_num or not f_num:
    st.error("숫자형 열이 없어요. 원본 데이터를 확인해보세요.")
    st.stop()

# 평균기온 열이 없고 TMIN, TMAX가 있으면 직접 계산
has_tavg = any("tavg" in c.lower() for c in w_num)
tmin = [c for c in w_num if "tmin" in c.lower()]
tmax = [c for c in w_num if "tmax" in c.lower()]
if not has_tavg and tmin and tmax:
    weather["W_TAVG(계산)"] = (weather[tmin[0]] + weather[tmax[0]]) / 2
    w_num = [c for c in weather.columns if c != "DATE"]

# ---------- 변수 선택 ----------
temp_col = st.sidebar.selectbox(
    "기온 변수", w_num,
    index=pick_default(w_num, ["tavg", "temp", "tmin", "tmax"]),
    format_func=label,
)
flu_col = st.sidebar.selectbox(
    "독감 의심 환자 지표", f_num,
    index=pick_default(f_num, ["suspect", "ili", "influenza", "flu", "case", "visit"]),
    format_func=label,
)
unit = st.sidebar.radio("기온 단위(원본 기준)", ["°F (화씨)", "°C (섭씨)"])
t_label = "°F" if unit.startswith("°F") else "°C"

# ---------- 합치기 ----------
exact = pd.merge(flu[["DATE", flu_col]], weather[["DATE", temp_col]], on="DATE", how="inner")
if len(exact) >= 0.8 * len(flu):
    merged = exact
    st.sidebar.success(f"날짜 정확히 일치: {len(merged)}행")
else:
    merged = pd.merge_asof(
        flu[["DATE", flu_col]], weather[["DATE", temp_col]],
        on="DATE", direction="nearest", tolerance=pd.Timedelta(days=3),
    )
    st.sidebar.warning("날짜가 안 맞아 ±3일 허용으로 합쳤어요.")

merged = merged.dropna().sort_values("DATE").reset_index(drop=True)
if len(merged) < 5:
    st.error("합쳐진 데이터가 너무 적어요. 날짜 열 선택을 확인해주세요.")
    st.write("날씨 날짜 범위:", weather["DATE"].min(), "~", weather["DATE"].max())
    st.write("인플루엔자 날짜 범위:", flu["DATE"].min(), "~", flu["DATE"].max())
    st.stop()

merged = merged.rename(columns={temp_col: "TEMP", flu_col: "FLU"})
merged["MONTH"] = merged["DATE"].dt.month

# 기간 선택
min_d, max_d = merged["DATE"].min().date(), merged["DATE"].max().date()
period = st.sidebar.date_input("기간 선택", value=(min_d, max_d), min_value=min_d, max_value=max_d)
if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("시작일과 종료일을 모두 선택해주세요.")
    st.stop()
merged = merged[
    (merged["DATE"].dt.date >= period[0]) & (merged["DATE"].dt.date <= period[1])
].reset_index(drop=True)
if len(merged) < 5:
    st.warning("선택한 기간의 데이터가 너무 적어요.")
    st.stop()

# ---------- 결과 요약 ----------
r = merged["TEMP"].corr(merged["FLU"])
st.subheader("📋 결과 요약")
c1, c2, c3 = st.columns(3)
c1.metric("분석한 주(행) 수", f"{len(merged):,}")
c2.metric("상관계수 r", "계산 불가" if pd.isna(r) else f"{r:.3f}")

median_t = merged["TEMP"].median()
cold = merged[merged["TEMP"] <= median_t]["FLU"].mean()
warm = merged[merged["TEMP"] > median_t]["FLU"].mean()
c3.metric("추운 주 평균 환자 / 따뜻한 주", f"{cold:,.1f} / {warm:,.1f}")

if pd.isna(r):
    st.info("상관계수를 계산할 수 없어요. 다른 변수를 선택해보세요.")
else:
    strength = "매우 약한" if abs(r) < 0.2 else "약한" if abs(r) < 0.4 else "뚜렷한" if abs(r) < 0.7 else "강한"
    if r < -0.2:
        st.success(f"✅ 기온이 낮을수록 환자가 많아지는 **{strength} 음의 상관**이 있어요. (r = {r:.3f})")
    elif r > 0.2:
        st.warning(f"⚠️ 오히려 기온이 높을수록 환자가 많아지는 **{strength} 양의 상관**이에요. (r = {r:.3f})")
    else:
        st.info(f"기온과 환자 수 사이에 뚜렷한 선형 관계가 없어요. (r = {r:.3f})")

with st.expander("합쳐진 데이터 보기"):
    st.dataframe(merged)

tab1, tab2, tab3, tab4 = st.tabs(["📈 시간 흐름", "🔵 산점도", "📊 기온 구간별", "⏱️ 시차 분석"])

# 1. 시간 흐름 (기온 축은 뒤집어서 '추워질수록 위로' 보이게 하는 옵션 제공)
with tab1:
    flip = st.checkbox("기온 축 뒤집기 (추울수록 위로)", value=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged["FLU"],
                             name=label(flu_col), line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged["TEMP"],
                             name=f"기온({t_label})", yaxis="y2",
                             line=dict(color="royalblue")))
    fig.update_layout(
        yaxis=dict(title=label(flu_col)),
        yaxis2=dict(title=f"기온({t_label})", overlaying="y", side="right",
                    autorange="reversed" if flip else True),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig)
    st.caption("빨간 선(환자)이 솟을 때 파란 선(기온)이 내려가 있는지 살펴보세요.")

# 2. 산점도
with tab2:
    fig = px.scatter(merged, x="TEMP", y="FLU", color="MONTH", opacity=0.6,
                     color_continuous_scale="Turbo",
                     labels={"TEMP": f"기온({t_label})", "FLU": label(flu_col)})
    if merged["TEMP"].std() > 0 and merged["FLU"].std() > 0:
        k, b = np.polyfit(merged["TEMP"], merged["FLU"], 1)
        xs = np.linspace(merged["TEMP"].min(), merged["TEMP"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="추세선",
                                 line=dict(color="black", dash="dash")))
        st.write(f"추세선 기울기: 기온이 1{t_label} 오를 때 환자 수는 평균 **{k:+.2f}** 변해요.")
    st.plotly_chart(fig)

# 3. 기온 구간별 평균 환자 수
with tab3:
    n_bins = st.slider("기온 구간 개수", 3, 10, 5)
    tmp = merged.copy()
    try:
        tmp["구간"] = pd.qcut(tmp["TEMP"], q=n_bins, duplicates="drop")
        g = tmp.groupby("구간", observed=True)["FLU"].agg(["mean", "count"]).reset_index()
        g["구간"] = g["구간"].astype(str)
        fig = px.bar(g, x="구간", y="mean", text="count",
                     labels={"구간": f"기온 구간({t_label}, 낮음 → 높음)",
                             "mean": f"평균 {label(flu_col)}"})
        fig.update_traces(texttemplate="%{text}주", textposition="outside")
        st.plotly_chart(fig)
        st.caption("막대 위 숫자는 그 구간에 속한 주의 개수예요. 왼쪽(추움)이 높고 오른쪽(따뜻함)이 낮으면 가설을 지지해요.")
    except ValueError:
        st.info("구간을 나눌 수 없어요. 구간 개수를 줄여보세요.")

# 4. 시차 분석
with tab4:
    st.markdown("기온이 변하고 **몇 주 뒤** 환자 수와 가장 강하게 연결되는지 봐요.")
    max_lag = st.slider("최대 시차(주)", 1, 12, 6)
    rows = [{"시차(주)": k, "상관계수": merged["FLU"].corr(merged["TEMP"].shift(k))}
            for k in range(max_lag + 1)]
    lag_df = pd.DataFrame(rows)
    fig = px.bar(lag_df, x="시차(주)", y="상관계수", range_y=[-1, 1])
    st.plotly_chart(fig)
    if lag_df["상관계수"].notna().any():
        best = lag_df.loc[lag_df["상관계수"].abs().idxmax()]
        st.success(f"가장 강한 시차: **{int(best['시차(주)'])}주 전 기온** (r = {best['상관계수']:.3f})")
