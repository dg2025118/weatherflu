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


# ---------- 데이터 불러오기 ----------
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

has_tavg = any("tavg" in c.lower() for c in w_num)
tmin = [c for c in w_num if "tmin" in c.lower()]
tmax = [c for c in w_num if "tmax" in c.lower()]
if not has_tavg and tmin and tmax:
    weather["W_TAVG(계산)"] = (weather[tmin[0]] + weather[tmax[0]]) / 2
    w_num = [c for c in weather.columns if c != "DATE"]

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

# ---------- 합치기 (값이 있는 행만) ----------
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

merged = merged.dropna(subset=[temp_col, flu_col]).sort_values("DATE").reset_index(drop=True)
if merged.empty:
    st.error("두 데이터가 겹치는 주가 없어요. 날짜 열 선택을 확인해주세요.")
    st.write("날씨 날짜 범위:", weather["DATE"].min(), "~", weather["DATE"].max())
    st.write("인플루엔자 날짜 범위:", flu["DATE"].min(), "~", flu["DATE"].max())
    st.stop()

merged = merged.rename(columns={temp_col: "TEMP", flu_col: "FLU"})
merged["YEAR"] = merged["DATE"].dt.year
merged["MONTH"] = merged["DATE"].dt.month

# 기간 선택
min_d, max_d = merged["DATE"].min().date(), merged["DATE"].max().date()
period = st.sidebar.date_input("기간 선택", value=(min
