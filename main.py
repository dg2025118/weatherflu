import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_weekly_averages_2009-10-03_to_2022-12-29.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="날씨 × 독감 조합 탐색", layout="wide")
st.title("🌡️ 날씨 × 독감: 겹치는 조합만 골라 보기")


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


def label(c):
    return c[2:] if c[:2] in ("W_", "F_") else c


def overview_table(df):
    return pd.DataFrame({
        "열 이름": df.columns,
        "자료형": df.dtypes.astype(str).values,
        "빈 값 개수": df.isna().sum().values,
    })


def iqr_mask(s, k):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


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
min_rows = st.sidebar.slider("조합에 필요한 최소 겹치는 주 수", 10, 200, 30, 5)
remove_out = st.sidebar.checkbox("이상치(IQR) 제거", value=False,
                                 help="독감 대유행 같은 진짜 극단값도 지워질 수 있어요.")
k_iqr = st.sidebar.slider("IQR 배수", 1.0, 3.0, 1.5, 0.5)

weather = prep(weather_raw, w_date, "W_")
flu = prep(flu_raw, f_date, "F_")

if weather.empty or flu.empty:
    st.error("날짜 열이 올바르지 않아요. 사이드바에서 다른 열을 선택해보세요.")
    st.stop()

# ---------- 1. 데이터 개요 ----------
st.header("1️⃣ 데이터 개요")
c1, c2 = st.columns(2)
with c1:
    st.subheader("🌤️ 날씨")
    a, b = st.columns(2)
    a.metric("행 개수", f"{len(weather_raw):,}")
    b.metric("열 개수", weather_raw.shape[1])
    st.dataframe(overview_table(weather_raw), hide_index=True)
with c2:
    st.subheader("🤧 인플루엔자")
    a, b = st.columns(2)
    a.metric("행 개수", f"{len(flu_raw):,}")
    b.metric("열 개수", flu_raw.shape[1])
    st.dataframe(overview_table(flu_raw), hide_index=True)

# ---------- 2. 쓸 수 없는 열 지우기 ----------
removed_cols = []


def usable(df, prefix_name):
    keep = []
    for c in df.columns:
        if c == "DATE":
            continue
        if df[c].notna().sum() == 0:
            removed_cols.append((prefix_name, label(c), "전부 비어 있음"))
        elif df[c].nunique(dropna=True) <= 1:
            removed_cols.append((prefix_name, label(c), "값이 하나뿐(변화 없음)"))
        else:
            keep.append(c)
    return keep


w_vars = usable(weather, "날씨")
f_vars = usable(flu, "인플루엔자")

if not w_vars or not f_vars:
    st.error("사용할 수 있는 열이 없어요. 원본 데이터를 확인해보세요.")
    st.stop()

# ---------- 합치기 ----------
exact = pd.merge(flu, weather, on="DATE", how="inner")
if len(exact) >= 0.8 * len(flu):
    merged = exact
    st.sidebar.success(f"날짜 정확히 일치: {len(merged)}행")
else:
    merged = pd.merge_asof(
        flu, weather, on="DATE", direction="nearest", tolerance=pd.Timedelta(days=3)
    )
    st.sidebar.warning("날짜가 안 맞아 ±3일 허용으로 합쳤어요.")
merged = merged.sort_values("DATE").reset_index(drop=True)
merged["YEAR"] = merged["DATE"].dt.year
merged["MONTH"] = merged["DATE"].dt.month

# ---------- 3. 가능한 조합 찾기 ----------
pairs, dropped_pairs = [], []
for w, f in itertools.product(w_vars, f_vars):
    sub = merged[["DATE", w, f]].dropna()
    n_before = len(sub)
    if remove_out and n_before >= 5:
        sub = sub[~(iqr_mask(sub[w], k_iqr) | iqr_mask(sub[f], k_iqr))]
    if len(sub) < min_rows:
        dropped_pairs.append({"날씨": label(w), "인플루엔자": label(f),
                              "겹치는 주": len(sub), "이유": f"{min_rows}주 미만"})
        continue
    if sub[w].std() == 0 or sub[f].std() == 0:
        dropped_pairs.append({"날씨": label(w), "인플루엔자": label(f),
                              "겹치는 주": len(sub), "이유": "값 변화 없음"})
        continue
    pairs.append({"w": w, "f": f, "n": len(sub), "n_before": n_before,
                  "r": sub[w].corr(sub[f])})

pairs.sort(key=lambda p: abs(p["r"]), reverse=True)

st.header("2️⃣ 전처리 결과: 겹치는 것만 남겼어요")
s1, s2, s3, s4 = st.columns(4)
s1.metric("가능한 전체 조합", len(w_vars) * len(f_vars))
s2.metric("남은 조합 ✅", len(pairs))
s3.metric("지워진 조합 ❌", len(dropped_pairs))
s4.metric("지워진 열", len(removed_cols))

with st.expander("지워진 열과 조합 보기"):
    if removed_cols:
        st.write("**지워진 열**")
        st.dataframe(pd.DataFrame(removed_cols, columns=["데이터", "열", "이유"]),
                     hide_index=True)
    if dropped_pairs:
        st.write("**지워진 조합**")
        st.dataframe(pd.DataFrame(dropped_pairs), hide_index=True)
    if not removed_cols and not dropped_pairs:
        st.success("지워진 것이 없어요.")

if not pairs:
    st.error("겹치는 주가 충분한 조합이 없어요. 사이드바에서 '최소 겹치는 주 수'를 줄여보세요.")
    st.write("날씨 날짜 범위:", weather["DATE"].min(), "~", weather["DATE"].max())
    st.write("인플루엔자 날짜 범위:", flu["DATE"].min(), "~", flu["DATE"].max())
    st.stop()

# 조합 요약 표
summary = pd.DataFrame([{
    "날씨 변수": label(p["w"]), "인플루엔자 지표": label(p["f"]),
    "처리 전 겹침(주)": p["n_before"], "처리 후(주)": p["n"], "상관계수 r": round(p["r"], 3),
} for p in pairs])
with st.expander("남은 조합 요약표 (|r| 큰 순서)"):
    st.dataframe(summary, hide_index=True)

# ---------- 4. 체크박스로 조합 선택 ----------
st.header("3️⃣ 보고 싶은 조합을 체크하세요")
st.caption("|r|이 큰 조합부터 나열했어요. 체크하면 아래에 그래프가 나타나요.")

selected = []
cols = st.columns(3)
for i, p in enumerate(pairs):
    txt = f"{label(p['w'])} ↔ {label(p['f'])}  (r={p['r']:.2f}, {p['n']}주)"
    with cols[i % 3]:
        if st.checkbox(txt, value=(i == 0), key=f"pair_{p['w']}_{p['f']}"):
            selected.append(p)

if not selected:
    st.info("조합을 하나 이상 체크해주세요.")
    st.stop()

# ---------- 5. 선택한 조합의 그래프 ----------
st.header("4️⃣ 선택한 조합의 그래프")

for p in selected:
    w, f = p["w"], p["f"]
    sub = merged[["DATE", "YEAR", "MONTH", w, f]].dropna().copy()
    if remove_out:
        sub = sub[~(iqr_mask(sub[w], k_iqr) | iqr_mask(sub[f], k_iqr))]
    sub = sub.rename(columns={w: "X", f: "Y"}).reset_index(drop=True)
    wl, fl = label(w), label(f)
    r = sub["X"].corr(sub["Y"])

    st.subheader(f"🔹 {wl} ↔ {fl}")
    m1, m2, m3 = st.columns(3)
    m1.metric("분석한 주", f"{len(sub):,} (처리 전 {p['n_before']:,})")
    m2.metric("상관계수 r", f"{r:.3f}")
    m3.metric("결정계수 R²", f"{r**2:.3f}")

    t1, t2, t3 = st.tabs(["📈 시간 흐름", "🔵 산점도", "📊 구간별 평균"])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["DATE"], y=sub["Y"], mode="lines",
                                 name=fl, line=dict(color="crimson", width=2)))
        fig.add_trace(go.Scatter(x=sub["DATE"], y=sub["X"], mode="lines",
                                 name=wl, yaxis="y2",
                                 line=dict(color="royalblue", width=1.5), opacity=0.7))
        fig.update_layout(
            title=f"{fl}와 {wl}의 시간 변화",
            xaxis=dict(title="날짜"),
            yaxis=dict(title=fl),
            yaxis2=dict(title=wl, overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12), hovermode="x unified", height=420,
        )
        st.plotly_chart(fig, key=f"line_{w}_{f}")
        st.info(f"🔎 {fl}의 최고값은 {sub.loc[sub['Y'].idxmax(), 'DATE'].date()}에 "
                f"{sub['Y'].max():,.1f}이고, 이때 {wl}은 {sub.loc[sub['Y'].idxmax(), 'X']:,.1f}예요.")

    with t2:
        fig = px.scatter(
            sub, x="X", y="Y", color="MONTH", color_continuous_scale="Turbo",
            hover_data={"DATE": True},
            title=f"{wl}와 {fl}의 관계 (점 하나 = 1주)",
            labels={"X": wl, "Y": fl, "MONTH": "월"},
        )
        k, b = np.polyfit(sub["X"], sub["Y"], 1)
        xs = np.linspace(sub["X"].min(), sub["X"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=k * xs + b, mode="lines", name="추세선",
                                 line=dict(color="black", dash="dash")))
        fig.update_layout(height=450)
        st.plotly_chart(fig, key=f"sc_{w}_{f}")
        direction = "음의 관계 (X가 클수록 Y 감소)" if r < -0.2 else (
            "양의 관계 (X가 클수록 Y 증가)" if r > 0.2 else "뚜렷한 선형 관계 없음")
        st.info(f"🔎 {direction}. 근거: r={r:.3f}, 기울기={k:+.3f}, {len(sub)}주.")

    with t3:
        n_bins = st.slider("구간 개수", 2, 8, 5, key=f"bins_{w}_{f}")
        try:
            tmp = sub.copy()
            tmp["구간"] = pd.qcut(tmp["X"], q=n_bins, duplicates="drop")
            g = tmp.groupby("구간", observed=True)["Y"].agg(["mean", "count"]).reset_index()
            g["구간"] = g["구간"].apply(lambda x: f"{x.left:.1f}~{x.right:.1f}")
            fig = px.bar(g, x="구간", y="mean", text="count",
                         title=f"{wl} 구간별 평균 {fl} (왼쪽 낮음 → 오른쪽 높음)",
                         labels={"구간": f"{wl} 구간", "mean": f"평균 {fl}"})
            fig.update_traces(texttemplate="%{text}주", textposition="outside")
            fig.update_layout(height=420)
            st.plotly_chart(fig, key=f"bar_{w}_{f}")
            st.info(f"🔎 가장 낮은 구간 평균 {g['mean'].iloc[0]:,.1f}, "
                    f"가장 높은 구간 평균 {g['mean'].iloc[-1]:,.1f}.")
        except ValueError:
            st.info("구간을 나눌 수 없어요. 구간 개수를 줄여보세요.")

    st.divider()
