import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_2009-10-03_to_2026-05-25.csv"

st.set_page_config(page_title="NYC 날씨 시각화", layout="wide")
st.title("🌦️ 뉴욕(NYC) 날씨 데이터 시각화 (2009~2026)")


@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df["TAVG"] = (df["TMIN"] + df["TMAX"]) / 2
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    return df


df = load_data(DATA_URL)

# ---------- 사이드바 ----------
st.sidebar.header("⚙️ 설정")

unit = st.sidebar.radio("온도 단위", ["°F (화씨)", "°C (섭씨)"])
rain_unit = st.sidebar.radio("강수/적설 단위", ["inch", "mm"])

min_d, max_d = df["DATE"].min().date(), df["DATE"].max().date()
period = st.sidebar.date_input(
    "기간 선택", value=(min_d, max_d), min_value=min_d, max_value=max_d
)
if len(period) != 2:
    st.info("시작일과 종료일을 모두 선택해주세요.")
    st.stop()
start, end = period

df = df[(df["DATE"].dt.date >= start) & (df["DATE"].dt.date <= end)].copy()

# 단위 변환
t_label = "°F"
if unit.startswith("°C"):
    for c in ["TMIN", "TMAX", "TAVG"]:
        df[c] = (df[c] - 32) * 5 / 9
    t_label = "°C"

r_label = "in"
if rain_unit == "mm":
    for c in ["PRCP", "SNOW", "SNWD"]:
        df[c] = df[c] * 25.4
    r_label = "mm"

if df.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

# ---------- 요약 지표 ----------
st.subheader("📋 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터 일수", f"{len(df):,}일")
c2.metric("평균 기온", f"{df['TAVG'].mean():.1f}{t_label}")
c3.metric("최고 기온 기록", f"{df['TMAX'].max():.1f}{t_label}")
c4.metric("최저 기온 기록", f"{df['TMIN'].min():.1f}{t_label}")

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

# ---------- 탭 ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🌡️ 기온 추이", "🌧️ 강수량", "❄️ 눈", "🗓️ 월별 패턴", "📅 연도별 추세", "🔥 상관관계"]
)

# 1. 기온 추이
with tab1:
    st.markdown("#### 일별 최저/최고 기온")
    window = st.slider("이동평균 (일)", 1, 60, 7)
    plot_df = df[["DATE", "TMIN", "TMAX"]].copy()
    plot_df["TMIN"] = plot_df["TMIN"].rolling(window, min_periods=1).mean()
    plot_df["TMAX"] = plot_df["TMAX"].rolling(window, min_periods=1).mean()
    fig = px.line(
        plot_df, x="DATE", y=["TMIN", "TMAX"],
        labels={"value": f"기온({t_label})", "variable": "구분"},
        color_discrete_map={"TMIN": "royalblue", "TMAX": "tomato"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 일교차 (TMAX - TMIN)")
    df["RANGE"] = df["TMAX"] - df["TMIN"]
    fig = px.histogram(df, x="RANGE", nbins=40,
                       labels={"RANGE": f"일교차({t_label})"})
    st.plotly_chart(fig, use_container_width=True)

# 2. 강수량
with tab2:
    st.markdown("#### 월별 총 강수량")
    monthly = df.groupby(df["DATE"].dt.to_period("M"))["PRCP"].sum().reset_index()
    monthly["DATE"] = monthly["DATE"].dt.to_timestamp()
    fig = px.bar(monthly, x="DATE", y="PRCP",
                 labels={"PRCP": f"강수량({r_label})", "DATE": "월"})
    st.plotly_chart(fig, use_container_width=True)

    rainy = (df["PRCP"] > 0).sum()
    st.write(f"🌧️ 비 온 날: **{rainy:,}일** ({rainy / len(df) * 100:.1f}%)")

    top = df.nlargest(10, "PRCP")[["DATE", "PRCP"]]
    st.markdown("#### 강수량 많았던 날 TOP 10")
    st.dataframe(top, use_container_width=True)

# 3. 눈
with tab3:
    st.markdown("#### 적설 깊이(SNWD)와 신적설량(SNOW)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["DATE"], y=df["SNWD"], name="적설 깊이(SNWD)",
                             fill="tozeroy"))
    fig.add_trace(go.Bar(x=df["DATE"], y=df["SNOW"], name="신적설(SNOW)"))
    fig.update_layout(yaxis_title=f"({r_label})")
    st.plotly_chart(fig, use_container_width=True)

    snow_year = df.groupby("YEAR")["SNOW"].sum().reset_index()
    st.markdown("#### 연도별 총 적설량")
    fig = px.bar(snow_year, x="YEAR", y="SNOW",
                 labels={"SNOW": f"적설량({r_label})"})
    st.plotly_chart(fig, use_container_width=True)

# 4. 월별 패턴
with tab4:
    st.markdown("#### 월별 평균 기온")
    m = df.groupby("MONTH")[["TMIN", "TAVG", "TMAX"]].mean().reset_index()
    fig = px.line(m, x="MONTH", y=["TMIN", "TAVG", "TMAX"], markers=True,
                  labels={"value": f"기온({t_label})", "MONTH": "월"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 월별 기온 분포 (박스플롯)")
    fig = px.box(df, x="MONTH", y="TAVG",
                 labels={"TAVG": f"평균기온({t_label})", "MONTH": "월"})
    st.plotly_chart(fig, use_container_width=True)

# 5. 연도별 추세
with tab5:
    st.markdown("#### 연도별 평균 기온")
    st.caption("※ 2009년과 2026년은 일부 기간만 있어 평균이 왜곡될 수 있어요.")
    y = df.groupby("YEAR")["TAVG"].mean().reset_index()
    fig = px.line(y, x="YEAR", y="TAVG", markers=True,
                  labels={"TAVG": f"평균기온({t_label})"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 연도 × 월 평균기온 히트맵")
    pivot = df.pivot_table(index="YEAR", columns="MONTH", values="TAVG", aggfunc="mean")
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="RdYlBu_r",
                    labels={"color": f"기온({t_label})", "x": "월", "y": "연도"})
    st.plotly_chart(fig, use_container_width=True)

# 6. 상관관계
with tab6:
    cols = ["PRCP", "SNOW", "SNWD", "TMIN", "TMAX", "TAVG"]
    corr = df[cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 산점도")
    cc1, cc2 = st.columns(2)
    x = cc1.selectbox("X축", cols, index=4)
    yv = cc2.selectbox("Y축", cols, index=0)
    fig = px.scatter(df, x=x, y=yv, opacity=0.4, color="MONTH",
                     color_continuous_scale="Turbo")
    st.plotly_chart(fig, use_container_width=True)
