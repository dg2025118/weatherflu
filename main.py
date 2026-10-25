import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_weekly_averages_2009-10-03_to_2022-12-29.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="NYC 날씨 & 인플루엔자", layout="wide")
st.title("🤧 뉴욕(NYC) 주간 날씨와 인플루엔자")


# ---------- 유틸 ----------
@st.cache_data
def load(url):
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_date_col(df):
    cols = list(df.columns)
    # 1순위: 이름에 date/week/day 포함 + 날짜로 잘 변환되는 열
    for key in ["date", "week", "day", "time"]:
        for c in cols:
            if key in c.lower():
                p = pd.to_datetime(df[c], errors="coerce")
                if p.notna().mean() > 0.8:
                    return c
    # 2순위: 문자열 열 중 날짜로 잘 변환되는 열
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            p = pd.to_datetime(df[c], errors="coerce")
            if p.notna().mean() > 0.8:
                return c
    return cols[0]


def prep(df, date_col, prefix):
    d = df.copy()
    d["DATE"] = pd.to_datetime(d[date_col], errors="coerce").dt.normalize()
    d = d.dropna(subset=["DATE"])
    d["DATE"] = d["DATE"].astype("datetime64[ns]")
    if date_col != "DATE":
        d = d.drop(columns=[date_col])

    # 쉼표 포함 문자 숫자 -> 숫자
    for c in d.columns:
        if c != "DATE" and not pd.api.types.is_numeric_dtype(d[c]):
            conv = pd.to_numeric(
                d[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            if conv.notna().mean() > 0.5:
                d[c] = conv

    num = [c for c in d.columns if c != "DATE" and pd.api.types.is_numeric_dtype(d[c])]
    d = d[["DATE"] + num]
    d = d.groupby("DATE", as_index=False).mean()      # 같은 날짜는 평균
    d = d.rename(columns={c: f"{prefix}{c}" for c in num})
    return d.sort_values("DATE").reset_index(drop=True)


weather_raw = load(WEATHER_URL)
flu_raw = load(FLU_URL)

# ---------- 사이드바 ----------
st.sidebar.header("⚙️ 설정")

w_default = guess_date_col(weather_raw)
f_default = guess_date_col(flu_raw)
w_date_col = st.sidebar.selectbox(
    "날씨 데이터의 날짜 열", weather_raw.columns,
    index=list(weather_raw.columns).index(w_default),
)
f_date_col = st.sidebar.selectbox(
    "인플루엔자 데이터의 날짜 열", flu_raw.columns,
    index=list(flu_raw.columns).index(f_default),
)

weather = prep(weather_raw, w_date_col, "W_")
flu = prep(flu_raw, f_date_col, "F_")

if weather.empty or flu.empty:
    st.error("날짜 열이 올바르지 않아요. 사이드바에서 다른 열을 선택해보세요.")
    st.write(weather_raw.head())
    st.write(flu_raw.head())
    st.stop()

w_num = [c for c in weather.columns if c != "DATE"]
f_num = [c for c in flu.columns if c != "DATE"]
if not w_num or not f_num:
    st.error("숫자형 열이 없어요. 원본 데이터를 확인해보세요.")
    st.write(weather_raw.head())
    st.write(flu_raw.head())
    st.stop()


def label(c):
    return c[2:]  # 접두어 제거해서 보여주기


w_cols = st.sidebar.multiselect(
    "분석할 날씨 변수", w_num, default=w_num[: min(3, len(w_num))], format_func=label
)
flu_col = st.sidebar.selectbox("분석할 인플루엔자 지표", f_num, format_func=label)

if not w_cols:
    st.info("사이드바에서 날씨 변수를 하나 이상 선택해주세요.")
    st.stop()

# ---------- 합치기 ----------
exact = pd.merge(flu[["DATE", flu_col]], weather[["DATE"] + w_cols], on="DATE", how="inner")

if len(exact) >= 0.8 * len(flu):
    merged = exact
    st.sidebar.success(f"날짜 정확히 일치: {len(merged)}행")
else:
    merged = pd.merge_asof(
        flu[["DATE", flu_col]].sort_values("DATE"),
        weather[["DATE"] + w_cols].sort_values("DATE"),
        on="DATE",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    ).dropna(subset=w_cols)
    st.sidebar.warning(f"날짜가 안 맞아 ±3일 허용으로 합침: {len(merged)}행")

merged = merged.dropna(subset=[flu_col]).reset_index(drop=True)
if merged.empty:
    st.error("합쳐진 데이터가 없어요. 두 파일의 날짜 열 선택을 확인해주세요.")
    st.write("날씨 날짜 범위:", weather["DATE"].min(), "~", weather["DATE"].max())
    st.write("인플루엔자 날짜 범위:", flu["DATE"].min(), "~", flu["DATE"].max())
    st.stop()

merged["YEAR"] = merged["DATE"].dt.year
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
if merged.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

# ---------- 요약 ----------
st.subheader("📋 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("합쳐진 행 수", f"{len(merged):,}")
c2.metric("기간", f"{period[0]} ~ {period[1]}")
c3.metric(f"{label(flu_col)} 평균", f"{merged[flu_col].mean():,.2f}")
c4.metric(f"{label(flu_col)} 최댓값", f"{merged[flu_col].max():,.2f}")

with st.expander("원본 날씨 데이터"):
    st.dataframe(weather_raw)
with st.expander("원본 인플루엔자 데이터"):
    st.dataframe(flu_raw)
with st.expander("합쳐진 데이터 (W_=날씨, F_=인플루엔자)"):
    st.dataframe(merged)
    st.caption(f"결측값 개수: {int(merged[[flu_col] + w_cols].isna().sum().sum())}")

# ---------- 탭 ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 함께 보기", "🔵 산점도", "🗓️ 계절 패턴", "⏱️ 시차 분석", "🔥 상관관계"]
)

# 1. 이중 축 시계열
with tab1:
    wcol = st.selectbox("함께 볼 날씨 변수", w_cols, format_func=label, key="t1")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged[flu_col],
                             name=label(flu_col), line=dict(color="crimson")))
    fig.add_trace(go.Scatter(x=merged["DATE"], y=merged[wcol], name=label(wcol),
                             yaxis="y2", line=dict(color="royalblue")))
    fig.update_layout(
        yaxis=dict(title=label(flu_col)),
        yaxis2=dict(title=label(wcol), overlaying="y", side="right"),
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    st.plotly_chart(fig)

# 2. 산점도
with tab2:
    wcol2 = st.selectbox("X축 날씨 변수", w_cols, format_func=label, key="t2")
    sub = merged[[wcol2, flu_col, "MONTH"]].dropna()
    fig = px.scatter(sub, x=wcol2, y=flu_col, color="MONTH", opacity=0.6,
                     color_continuous_scale="Turbo",
                     labels={wcol2: label(wcol2), flu_col: label(flu_col)})
    if len(sub) > 2 and sub[wcol2].std() > 0 and sub[flu_col].std() > 0:
        k, b = np.polyfit(sub[wcol2], sub[flu_col], 1)
        xs = np.linspace(sub[wcol2].min(), sub[wcol2].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="추세선",
                                 line=dict(color="black", dash="dash")))
        st.write(f"상관계수 r = **{sub[wcol2].corr(sub[flu_col]):.3f}**")
    else:
        st.info("값의 변화가 없거나 데이터가 부족해서 추세선을 그릴 수 없어요.")
    st.plotly_chart(fig)

# 3. 계절 패턴
with tab3:
    wcol3 = st.selectbox("비교할 날씨 변수", w_cols, format_func=label, key="t3")
    m = merged.groupby("MONTH")[[flu_col, wcol3]].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["MONTH"], y=m[flu_col], name=label(flu_col),
                         marker_color="crimson", opacity=0.7))
    fig.add_trace(go.Scatter(x=m["MONTH"], y=m[wcol3], name=label(wcol3), yaxis="y2",
                             mode="lines+markers", line=dict(color="royalblue")))
    fig.update_layout(
        xaxis=dict(title="월", dtick=1),
        yaxis=dict(title=label(flu_col)),
        yaxis2=dict(title=label(wcol3), overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig)

    st.markdown("#### 연도 × 월 히트맵")
    pivot = merged.pivot_table(index="YEAR", columns="MONTH", values=flu_col, aggfunc="mean")
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                    labels={"x": "월", "y": "연도", "color": label(flu_col)})
    st.plotly_chart(fig)

# 4. 시차 분석
with tab4:
    st.markdown("날씨가 변하고 **몇 주 뒤**에 인플루엔자가 반응하는지 확인해요. (1칸 = 1주)")
    wcol4 = st.selectbox("날씨 변수", w_cols, format_func=label, key="t4")
    max_lag = st.slider("최대 시차(주)", 1, 20, 8)
    rows = []
    for k in range(max_lag + 1):
        r = merged[flu_col].corr(merged[wcol4].shift(k))
        rows.append({"시차(주)": k, "상관계수": r})
    lag_df = pd.DataFrame(rows)
    fig = px.bar(lag_df, x="시차(주)", y="상관계수", range_y=[-1, 1])
    st.plotly_chart(fig)
    if lag_df["상관계수"].notna().any():
        best = lag_df.loc[lag_df["상관계수"].abs().idxmax()]
        st.success(f"절댓값이 가장 큰 시차: **{int(best['시차(주)'])}주** (r = {best['상관계수']:.3f})")
    else:
        st.info("상관계수를 계산할 수 없어요. 다른 변수를 선택해보세요.")

# 5. 상관관계
with tab5:
    corr = merged[[flu_col] + w_cols].corr()
    corr.index = [label(c) for c in corr.index]
    corr.columns = [label(c) for c in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig)
