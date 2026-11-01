import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

WEATHER_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_weather_weekly_averages_2009-10-03_to_2022-12-29.csv"
FLU_URL = "https://raw.githubusercontent.com/dg2025118/weatherflu/refs/heads/main/NYC_influenza_2009-10-03_to_2022-12-29%20(4).csv"

st.set_page_config(page_title="기온과 독감 의심 환자", layout="wide")
st.title("🌡️ 기온이 낮아질수록 독감 의심 환자가 많아질까?")
st.caption("뉴욕시(NYC) 주간 기온 데이터와 인플루엔자 의심 환자 데이터 분석")


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


def overview_table(df):
    return pd.DataFrame({
        "열 이름": df.columns,
        "자료형": df.dtypes.astype(str).values,
        "빈 값 개수": df.isna().sum().values,
    })


def iqr_mask(s, k):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi), lo, hi


def interpretation_box(key):
    st.text_area("✍️ 우리 팀의 해석 (왜 이런 결과가 나왔을까?)", key=key,
                 placeholder="예: 겨울에 환자가 늘어난 이유는 ...")


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
    st.stop()

w_num = [c for c in weather.columns if c != "DATE"]
f_num = [c for c in flu.columns if c != "DATE"]
if not w_num or not f_num:
    st.error("숫자형 열이 없어요. 원본 데이터를 확인해보세요.")
    st.stop()

if not any("tavg" in c.lower() for c in w_num):
    tmin = [c for c in w_num if "tmin" in c.lower()]
    tmax = [c for c in w_num if "tmax" in c.lower()]
    if tmin and tmax:
        weather["W_TAVG(계산)"] = (weather[tmin[0]] + weather[tmax[0]]) / 2
        w_num = [c for c in weather.columns if c != "DATE"]

temp_col = st.sidebar.selectbox(
    "기온 변수", w_num,
    index=pick_default(w_num, ["tavg", "temp", "tmin", "tmax"]), format_func=label,
)
flu_col = st.sidebar.selectbox(
    "독감 의심 환자 지표", f_num,
    index=pick_default(f_num, ["suspect", "ili", "influenza", "flu", "case", "visit"]),
    format_func=label,
)
unit = st.sidebar.radio("기온 단위(원본 기준)", ["°F (화씨)", "°C (섭씨)"])
t_label = "°F" if unit.startswith("°F") else "°C"

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 전처리 방법")
st.sidebar.write("결측치: 기온 또는 환자 수가 비어 있는 주는 **삭제**")
outlier_mode = st.sidebar.radio(
    "이상치 처리", ["유지 (표시만 함)", "제거"],
    help="독감 대유행 같은 진짜 극단값은 지우면 안 될 수도 있어요.",
)
k_iqr = st.sidebar.slider("IQR 배수", 1.0, 3.0, 1.5, 0.5)

# ---------- 합치기 ----------
exact = pd.merge(flu[["DATE", flu_col]], weather[["DATE", temp_col]], on="DATE", how="inner")
if len(exact) >= 0.8 * len(flu):
    merged_all = exact
    st.sidebar.success(f"날짜 정확히 일치: {len(merged_all)}행")
else:
    merged_all = pd.merge_asof(
        flu[["DATE", flu_col]], weather[["DATE", temp_col]],
        on="DATE", direction="nearest", tolerance=pd.Timedelta(days=3),
    )
    st.sidebar.warning("날짜가 안 맞아 ±3일 허용으로 합쳤어요.")

merged_all = merged_all.rename(columns={temp_col: "TEMP", flu_col: "FLU"})
merged_all = merged_all.sort_values("DATE").reset_index(drop=True)

# 단계별 처리
n_flu_raw = len(flu_raw)
n_merged = len(merged_all)
clean = merged_all.dropna(subset=["TEMP", "FLU"]).reset_index(drop=True)
n_after_missing = len(clean)

if clean.empty:
    st.error("두 데이터가 겹치는 주가 없어요. 날짜 열 선택을 확인해주세요.")
    st.write("날씨 날짜 범위:", weather["DATE"].min(), "~", weather["DATE"].max())
    st.write("인플루엔자 날짜 범위:", flu["DATE"].min(), "~", flu["DATE"].max())
    st.stop()

m_t, t_lo, t_hi = iqr_mask(clean["TEMP"], k_iqr)
m_f, f_lo, f_hi = iqr_mask(clean["FLU"], k_iqr)
clean["이상치"] = np.where(m_t & m_f, "기온+환자", np.where(m_t, "기온", np.where(m_f, "환자 수", "정상")))
n_outliers = int((m_t | m_f).sum())

if outlier_mode == "제거":
    data = clean[~(m_t | m_f)].reset_index(drop=True)
else:
    data = clean.copy()
n_final = len(data)

if len(data) < 5:
    st.error("분석할 데이터가 너무 적어요. 이상치 처리 방법을 '유지'로 바꿔보세요.")
    st.stop()

data["YEAR"] = data["DATE"].dt.year
data["MONTH"] = data["DATE"].dt.month

# =====================================================
# 1️⃣ 데이터 개요
# =====================================================
st.header("1️⃣ 데이터 개요")
oc1, oc2 = st.columns(2)
with oc1:
    st.subheader("🌤️ 날씨 데이터")
    a, b = st.columns(2)
    a.metric("행 개수", f"{len(weather_raw):,}")
    b.metric("열 개수", f"{weather_raw.shape[1]}")
    st.dataframe(overview_table(weather_raw), hide_index=True)
with oc2:
    st.subheader("🤧 인플루엔자 데이터")
    a, b = st.columns(2)
    a.metric("행 개수", f"{len(flu_raw):,}")
    b.metric("열 개수", f"{flu_raw.shape[1]}")
    st.dataframe(overview_table(flu_raw), hide_index=True)

with st.expander("원본 데이터 미리보기"):
    st.write("날씨", weather_raw.head(10))
    st.write("인플루엔자", flu_raw.head(10))

# =====================================================
# 2️⃣ 결측치·이상치 처리
# =====================================================
st.header("2️⃣ 결측치·이상치 처리")
st.markdown(
    f"- **결측치:** 기온 또는 환자 수가 비어 있는 주는 **삭제**했어요.\n"
    f"- **이상치:** IQR 방법(Q1 − {k_iqr}×IQR 미만, Q3 + {k_iqr}×IQR 초과)으로 찾았고, "
    f"현재 설정은 **{outlier_mode}**이에요."
)

steps = pd.DataFrame({
    "단계": ["① 인플루엔자 원본", "② 날씨와 합친 후", "③ 결측치 삭제 후", "④ 이상치 처리 후(최종)"],
    "행 수": [n_flu_raw, n_merged, n_after_missing, n_final],
})
steps["이전 단계 대비 변화"] = steps["행 수"].diff().fillna(0).astype(int)

sc1, sc2 = st.columns([1, 1])
with sc1:
    st.dataframe(steps, hide_index=True)
    st.write(
        f"**처리 전 {n_merged}행 → 처리 후 {n_final}행** "
        f"(결측 {n_merged - n_after_missing}개 삭제, 이상치 {n_outliers}개 발견"
        f"{' → 제거' if outlier_mode == '제거' else ' → 유지'})"
    )
    st.write(f"기온 정상 범위: {t_lo:,.1f} ~ {t_hi:,.1f}{t_label} / 이상치 {int(m_t.sum())}개")
    st.write(f"환자 수 정상 범위: {f_lo:,.1f} ~ {f_hi:,.1f} / 이상치 {int(m_f.sum())}개")
with sc2:
    fig = px.bar(steps, x="단계", y="행 수", text="행 수",
                 title="처리 단계별 행 수 변화",
                 labels={"단계": "처리 단계", "행 수": "행(주) 개수"})
    fig.update_traces(textposition="outside")
    fig.update_layout(height=330)
    st.plotly_chart(fig)

bc1, bc2 = st.columns(2)
for col, var, title in [(bc1, "TEMP", f"기온({t_label}) 상자그림"), (bc2, "FLU", f"{label(flu_col)} 상자그림")]:
    with col:
        fig = px.box(clean, y=var, points="outliers", hover_data=["DATE"],
                     title=title, labels={var: title.replace(" 상자그림", "")})
        fig.update_layout(height=340)
        st.plotly_chart(fig)

out_df = clean[clean["이상치"] != "정상"]
with st.expander(f"이상치로 판정된 주 보기 ({len(out_df)}개)"):
    if out_df.empty:
        st.success("이상치가 없어요.")
    else:
        st.dataframe(out_df[["DATE", "TEMP", "FLU", "이상치"]].rename(
            columns={"TEMP": f"기온({t_label})", "FLU": label(flu_col)}), hide_index=True)
        st.caption("환자 수 이상치가 겨울에 몰려 있다면 오류가 아니라 실제 독감 유행일 수 있어요.")

# =====================================================
# 3️⃣ 질문과 연결된 그래프
# =====================================================
st.header("3️⃣ 우리 팀의 질문: 기온이 낮아질수록 독감 의심 환자가 많아질까?")

r = data["TEMP"].corr(data["FLU"])
st.metric("전체 상관계수 r (음수일수록 '추울수록 환자 많음')", "계산 불가" if pd.isna(r) else f"{r:.3f}")

# ---- 그래프 1 ----
st.subheader("📈 그래프 1. 시간에 따른 환자 수와 기온")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["DATE"], y=data["FLU"], mode="lines",
                         name=label(flu_col), line=dict(color="crimson", width=2)))
fig.add_trace(go.Scatter(x=data["DATE"], y=data["TEMP"], mode="lines",
                         name=f"기온({t_label}, 위쪽이 추움)", yaxis="y2",
                         line=dict(color="royalblue", width=1.5), opacity=0.7))
fig.update_layout(
    title="주간 독감 의심 환자 수와 기온의 변화 (기온 축은 뒤집힘)",
    xaxis=dict(title="날짜"),
    yaxis=dict(title=f"독감 의심 환자 수 ({label(flu_col)})"),
    yaxis2=dict(title=f"기온({t_label}) ↑ 추움", overlaying="y", side="right", autorange="reversed"),
    legend=dict(orientation="h", y=1.12), hovermode="x unified", height=450,
)
st.plotly_chart(fig)

top_cut = data["FLU"].quantile(0.9)
top_weeks = data[data["FLU"] >= top_cut]
top_temp = top_weeks["TEMP"].mean()
all_temp = data["TEMP"].mean()
peak_months = top_weeks["MONTH"].value_counts().head(3)
months_txt = ", ".join(f"{m}월({n}주)" for m, n in peak_months.items())
st.info(
    f"🔎 **찾아낸 것:** 환자가 가장 많은 상위 10% 주의 평균 기온은 {top_temp:.1f}{t_label}로, "
    f"전체 평균 {all_temp:.1f}{t_label}보다 {'낮아요' if top_temp < all_temp else '낮지 않아요'}.\n\n"
    f"📊 **근거 수치:** 상위 10% 주 {len(top_weeks)}개 중 가장 많은 달은 {months_txt}."
)
interpretation_box("note1")

# ---- 그래프 2 ----
st.subheader("🔵 그래프 2. 기온과 환자 수의 관계 (산점도)")
fig = px.scatter(
    data, x="TEMP", y="FLU", color="MONTH", color_continuous_scale="Turbo",
    hover_data={"DATE": True, "TEMP": ":.1f", "FLU": ":.1f"},
    title="기온과 독감 의심 환자 수의 관계 (점 하나 = 1주)",
    labels={"TEMP": f"주간 평균 기온({t_label})", "FLU": f"독감 의심 환자 수 ({label(flu_col)})",
            "MONTH": "월"},
)
slope = np.nan
if data["TEMP"].std() > 0 and data["FLU"].std() > 0:
    slope, intercept = np.polyfit(data["TEMP"], data["FLU"], 1)
    xs = np.linspace(data["TEMP"].min(), data["TEMP"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="추세선",
                             line=dict(color="black", dash="dash")))
fig.update_layout(height=470)
st.plotly_chart(fig)

if pd.isna(r):
    st.info("상관계수를 계산할 수 없어요.")
else:
    direction = "기온이 낮을수록 환자가 많아지는 경향" if r < -0.2 else (
        "기온이 높을수록 환자가 많아지는 경향" if r > 0.2 else "뚜렷한 선형 관계 없음")
    strength = "매우 약함" if abs(r) < 0.2 else "약함" if abs(r) < 0.4 else "뚜렷함" if abs(r) < 0.7 else "강함"
    st.info(
        f"🔎 **찾아낸 것:** {direction} ({strength}).\n\n"
        f"📊 **근거 수치:** 상관계수 r = {r:.3f}, 결정계수 R² = {r**2:.3f}, "
        f"추세선 기울기 = 기온 1{t_label} 상승 시 환자 수 {slope:+.2f} 변화, 분석한 주 {len(data)}개."
    )
interpretation_box("note2")

# ---- 그래프 3 ----
st.subheader("📊 그래프 3. 기온 구간별 평균 환자 수")
max_bins = max(2, min(8, len(data) // 2))
n_bins = st.slider("기온 구간 개수", 2, max_bins, min(5, max_bins))
try:
    tmp = data.copy()
    tmp["구간"] = pd.qcut(tmp["TEMP"], q=n_bins, duplicates="drop")
    g = tmp.groupby("구간", observed=True)["FLU"].agg(["mean", "count"]).reset_index()
    g["구간"] = g["구간"].apply(lambda x: f"{x.left:.0f}~{x.right:.0f}{t_label}")
    g["순서"] = range(len(g))
    fig = px.bar(
        g, x="구간", y="mean", text="count", color="순서", color_continuous_scale="RdYlBu_r",
        title="기온 구간별 평균 독감 의심 환자 수 (왼쪽 추움 → 오른쪽 따뜻함)",
        labels={"구간": f"주간 평균 기온 구간({t_label})", "mean": f"평균 환자 수 ({label(flu_col)})"},
    )
    fig.update_traces(texttemplate="%{text}주", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, height=450)
    st.plotly_chart(fig)

    first, last = g["mean"].iloc[0], g["mean"].iloc[-1]
    ratio = first / last if last else np.nan
    st.info(
        f"🔎 **찾아낸 것:** 가장 추운 구간의 환자 수가 가장 따뜻한 구간보다 "
        f"{'많아요' if first > last else '많지 않아요'}.\n\n"
        f"📊 **근거 수치:** 가장 추운 구간({g['구간'].iloc[0]}) 평균 {first:,.1f}, "
        f"가장 따뜻한 구간({g['구간'].iloc[-1]}) 평균 {last:,.1f}"
        + (f", 비율 {ratio:.1f}배." if not pd.isna(ratio) else ".")
    )
except ValueError:
    st.info("구간을 나눌 수 없어요. 구간 개수를 줄여보세요.")
interpretation_box("note3")

# =====================================================
# 마무리
# =====================================================
st.header("✅ 결론")
st.text_area("팀의 최종 결론과 한계점 (구술 때 이 내용을 말해요)", key="final",
             placeholder="예: 기온이 낮을수록 환자가 많은 경향이 있었다(r=...). 하지만 상관관계가 인과관계는 아니고 ...")

with st.expander("분석에 사용된 최종 데이터"):
    st.dataframe(data)
