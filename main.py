import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_2009-10-03_to_2026-05-25.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="NYC 날씨 & 인플루엔자", layout="wide")
st.title("🤧 뉴욕(NYC) 인플루엔자 기준 날씨 분석")


# ---------- 데이터 불러오기 ----------
@st.cache_data
def load_weather(url):
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
    df["TAVG"] = (df["TMIN"] + df["TMAX"]) / 2
    return df


@st.cache_data
def load_flu(url):
    return pd.read_csv(url)


weather = load_weather(WEATHER_URL)
flu_raw = load_flu(FLU_URL)

# ---------- 사이드바 ----------
st.sidebar.header("⚙️ 설정")

guess = [c for c in flu_raw.columns if "date" in c.lower()]
default_idx = list(flu_raw.columns).index(guess[0]) if guess else 0
flu_date_col = st.sidebar.selectbox(
    "인플루엔자 데이터의 날짜 열", flu_raw.columns, index=default_idx
)

date_meaning = st.sidebar.radio(
    "인플루엔자 날짜의 의미",
    ["구간의 끝 (그 날짜 이전 기간을 대표)", "구간의 시작 (그 날짜 이후 기간을 대표)"],
)
temp_unit = st.sidebar.radio("온도 단위", ["°F (화씨)", "°C (섭씨)"])

# ---------- 인플루엔자 전처리 ----------
flu = flu_raw.copy()
flu[flu_date_col] = pd.to_datetime(flu[flu_date_col], errors="coerce")
flu = flu.dropna(subset=[flu_date_col]).rename(columns={flu_date_col: "DATE"})

if flu.empty:
    st.error("날짜 열이 올바르지 않아요. 사이드바에서 다른 열을 선택해보세요.")
    st.dataframe(flu_raw.head(20))
    st.stop()

# 쉼표 포함 숫자(예: '1,234')를 숫자로 변환 시도
for c in flu.columns:
    if c != "DATE" and flu[c].dtype == object:
        converted = pd.to_numeric(
            flu[c].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
        if converted.notna().mean() > 0.5:
            flu[c] = converted

flu = flu.sort_values("DATE").drop_duplicates(subset="DATE").reset_index(drop=True)

flu_num_cols = flu.select_dtypes(include="number").columns.tolist()
if not flu_num_cols:
    st.error("인플루엔자 데이터에 숫자형 열이 없어요. 원본을 확인해보세요.")
    st.dataframe(flu_raw.head(20))
    st.stop()

flu_col = st.sidebar.selectbox("분석할 인플루엔자 지표", flu_num_cols)

# ---------- 날씨를 인플루엔자 날짜 구간으로 집계 ----------
flu_dates = flu["DATE"].to_numpy(dtype="datetime64[ns]")
w = weather.copy()
w_dates = w["DATE"].to_numpy(dtype="datetime64[ns]")

if date_meaning.startswith("구간의 끝"):
    # 날씨 날짜 이후 가장 가까운 인플루엔자 날짜에 배정
    idx = np.searchsorted(flu_dates, w_dates, side="left")
else:
    # 날씨 날짜 이전(같은 날 포함) 가장 가까운 인플루엔자 날짜에 배정
    idx = np.searchsorted(flu_dates, w_dates, side="right") - 1

valid = (idx >= 0) & (idx < len(flu_dates))
w = w[valid].copy()
w["DATE_KEY"] = flu_dates[idx[valid]]

# 구간 길이가 너무 길게 벌어진 날(데이터 공백)은 제외: 최대 14일
gap_days = np.abs((w["DATE_KEY"].to_numpy() - w["DATE"].to_numpy()) / np.timedelta64(1, "D"))
w = w[gap_days <= 14]

w_agg = (
    w.groupby("DATE_KEY")
    .agg(
        TMIN=("TMIN", "mean"),
        TMAX=("TMAX", "mean"),
        TAVG=("TAVG", "mean"),
        PRCP=("PRCP", "sum"),
        SNOW=("SNOW", "sum"),
        SNWD=("SNWD", "mean"),
        WEATHER_DAYS=("DATE", "count"),
    )
    .reset_index()
    .rename(columns={"DATE_KEY": "DATE"})
)

merged = flu.merge(w_agg, on="DATE", how="inner")

if merged.empty:
    st.error("합쳐진 데이터가 없어요. 날짜 열 선택을 확인해주세요.")
    st.stop()

# 단위 변환
t_label = "°F"
if temp_unit.startswith("°C"):
    for c in ["TMIN", "TMAX", "TAVG"]:
        merged[c] = (merged[c] - 32) * 5 / 9
    t_label = "°C"

merged["YEAR"] = merged["DATE"].dt.year
merged["MONTH"] = merged["DATE"].dt.month

# 기간 선택
min_d, max_d = merged["DATE"].min().date(), merged["DATE"].max().date()
period = st.sidebar.date_input(
    "기간 선택", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if len(period) != 2:
    st.info("시작일과 종료일을 모두 선택해주세요.")
    st.stop()

merged = merged[
    (merged["DATE"].dt.date >= period[0]) & (merged["DATE"].dt.date <= period[1])
].copy()

if merged.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

weather_cols = ["TMIN", "TMAX", "TAVG", "PRCP", "SNOW", "SNWD"]
weather_names = {
    "TMIN": f"최저기온 평균({t_label})",
    "TMAX": f"최고기온 평균({t_label})",
    "TAVG": f"평균기온({t_label})",
    "PRCP": "강수량 합계(in)",
    "SNOW": "신적설 합계(in)",
    "SNWD": "적설깊이 평균(in)",
}

# ---------- 요약 ----------
st.subheader("📋 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("합쳐진 행 수", f"{len(merged):,}")
c2.metric("기간", f"{period[0]} ~ {period[1]}")
c3.metric(f"{flu_col} 평균", f"{merged[flu_col].mean():,.2f}")
c4.metric(f"{flu_col} 최댓값", f"{merged[flu_col].max():,.2f}")

diffs = flu["DATE"].diff().dt.days.dropna()
if not diffs.empty:
    st.caption(f"인플루엔자 데이터의 날짜 간격(가장 흔한 값): {int(diffs.mode().iloc[0])}일")

with st.expander("원본 인플루엔자 데이터 보기"):
    st.dataframe(flu_raw, use_container_width=True)
with st.expander("합쳐진 데이터 보기"):
    st.dataframe(merged, use_container_width=True)
    st.caption(
        f"결측값 개수: {int(merged[[flu_col] + weather_cols].isna().sum().sum())} / "
        "WEATHER_DAYS = 그 구간에 포함된 날씨 데이터 일수"
    )

# ---------- 탭 ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 함께 보기", "🔵 산점도", "🗓️ 계절 패턴", "⏱️ 시차 분석", "🔥 상관관계"]
)

# 1. 이중 축 시계열
with tab1:
    wcol = st.selectbox(
        "함께 볼 날씨 변수", weather_cols,
        format_func=lambda c: weather_names[c], index=2,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged[flu_col],
                             name=flu_col, line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged[wcol],
                             name=weather_names[wcol], yaxis="y2",
                             line=dict(color="royalblue")))
    fig.update_layout(
        yaxis=dict(title=flu_col),
        yaxis2=dict(title=weather_names[wcol], overlaying="y", side="right"),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. 산점도
with tab2:
    wcol2 = st.selectbox(
        "X축 날씨 변수", weather_cols,
        format_func=lambda c: weather_names[c], index=2, key="sc",
    )
    sub = merged[[wcol2, flu_col, "MONTH"]].dropna()
    fig = px.scatter(sub, x=wcol2, y=flu_col, color="MONTH", opacity=0.6,
                     color_continuous_scale="Turbo",
                     labels={wcol2: weather_names[wcol2]})
    if len(sub) > 2:
        k, b = np.polyfit(sub[wcol2], sub[flu_col], 1)
        xs = np.linspace(sub[wcol2].min(), sub[wcol2].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines",
                                 name="추세선", line=dict(color="black", dash="dash")))
        r = sub[wcol2].corr(sub[flu_col])
        st.write(f"상관계수 r = **{r:.3f}**")
    st.plotly_chart(fig, use_container_width=True)

# 3. 계절 패턴
with tab3:
    st.markdown("#### 월별 평균")
    m = merged.groupby("MONTH")[[flu_col, "TAVG"]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["MONTH"], y=m[flu_col], name=flu_col,
                         marker_color="crimson", opacity=0.7))
    fig.add_trace(go.Scatter(x=m["MONTH"], y=m["TAVG"], name=f"평균기온({t_label})",
                             yaxis="y2", mode="lines+markers",
                             line=dict(color="royalblue")))
    fig.update_layout(
        xaxis=dict(title="월", dtick=1),
        yaxis=dict(title=flu_col),
        yaxis2=dict(title=f"평균기온({t_label})", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 연도 × 월 히트맵")
    pivot = merged.pivot_table(index="YEAR", columns="MONTH",
                               values=flu_col, aggfunc="mean")
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                    labels={"x": "월", "y": "연도", "color": flu_col})
    st.plotly_chart(fig, use_container_width=True)

# 4. 시차 분석
with tab4:
    st.markdown("날씨가 변한 뒤 **몇 칸(행) 뒤**에 인플루엔자가 반응하는지 확인해요.")
    st.caption("데이터가 주 단위라면 1칸 = 1주예요.")
    wcol3 = st.selectbox(
        "날씨 변수", weather_cols,
        format_func=lambda c: weather_names[c], index=2, key="lag",
    )
    max_lag = st.slider("최대 시차(칸)", 1, 20, 8)
    rows = []
    for k in range(0, max_lag + 1):
        r = merged[flu_col].corr(merged[wcol3].shift(k))
        rows.append({"시차": k, "상관계수": r})
    lag_df = pd.DataFrame(rows)
    fig = px.bar(lag_df, x="시차", y="상관계수", range_y=[-1, 1])
    st.plotly_chart(fig, use_container_width=True)
    if lag_df["상관계수"].notna().any():
        best = lag_df.loc[lag_df["상관계수"].abs().idxmax()]
        st.success(f"절댓값이 가장 큰 시차: **{int(best['시차'])}칸** (r = {best['상관계수']:.3f})")

# 5. 상관관계
with tab5:
    cols = [flu_col] + weather_cols
    corr = merged[cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
