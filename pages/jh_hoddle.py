import calendar
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="텅장 구조대", layout="wide")

DEFAULT_LEDGER = "sample_household_ledger.csv"
DEFAULT_BUDGET = "sample_monthly_budget.csv"

def safe_read_csv(path_or_file):
    if path_or_file is None:
        return None
    if hasattr(path_or_file, "read"):
        try:
            return pd.read_csv(path_or_file, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path_or_file)
    p = Path(path_or_file)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(p)

def normalize_columns_ledger(df):
    df = df.copy()
    cols = {c: str(c).strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    required = ["날짜", "시간", "카테고리", "목적", "비용", "비고"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"가계부 CSV에 필요한 컬럼이 없습니다: {missing}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"])
    df["시간"] = df["시간"].astype(str).str.strip()
    df["카테고리"] = df["카테고리"].astype(str).str.strip()
    df["목적"] = df["목적"].astype(str).str.strip()
    df["비고"] = df["비고"].astype(str).fillna("").str.strip()

    df["비용"] = (
        df["비용"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["비용"] = pd.to_numeric(df["비용"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["날짜", "시간"]).reset_index(drop=True)

    def parse_hour(x):
        s = str(x)
        if ":" in s:
            try:
                return int(s.split(":")[0])
            except Exception:
                return np.nan
        return np.nan

    df["hour"] = df["시간"].apply(parse_hour)
    df["hour"] = df["hour"].fillna(12).astype(int)
    df["weekday"] = df["날짜"].dt.weekday
    df["weekday_name"] = df["weekday"].map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})
    df["month"] = df["날짜"].dt.to_period("M").astype(str)
    return df

def normalize_columns_budget(df):
    df = df.copy()
    cols = {c: str(c).strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    required = ["month", "salary", "budget", "payday"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"월급/예산 CSV에 필요한 컬럼이 없습니다: {missing}")

    df["month"] = df["month"].astype(str).str.strip()
    for c in ["salary", "budget", "payday"]:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["salary"] = df["salary"].fillna(0).astype(int)
    df["budget"] = df["budget"].fillna(0).astype(int)
    df["payday"] = df["payday"].fillna(25).astype(int)
    df = df.sort_values("month").reset_index(drop=True)
    return df

def assign_reinterpreted_category(df, impulsive_threshold, lazy_keywords, impulsive_keywords):
    df = df.copy()

    base_breath = set(["주거", "통신", "보험", "구독"])
    base_growth = set(["교육"])

    def contains_any(text, keywords):
        t = str(text)
        return any(k in t for k in keywords)

    def classify(row):
        cat = row["카테고리"]
        purpose = row["목적"]
        memo = row["비고"]
        cost = row["비용"]

        if cat in base_breath:
            return "호흡"
        if cat in base_growth:
            return "성장"
        if "헬스장" in purpose or "운동" in purpose:
            return "성장"

        lazy_hit = False
        if cat == "교통":
            if "택시" in memo or "대리" in memo or "지각" in purpose or "피곤" in purpose:
                lazy_hit = True
        if cat == "식사":
            if "배달" in purpose or "배달" in memo or "배달앱" in memo:
                lazy_hit = True
        if contains_any(purpose, lazy_keywords) or contains_any(memo, lazy_keywords):
            lazy_hit = True
        if lazy_hit:
            return "게으름"

        impulsive_hit = False
        if cat == "쇼핑" and cost >= impulsive_threshold:
            impulsive_hit = True
        if contains_any(purpose, impulsive_keywords) or contains_any(memo, impulsive_keywords):
            impulsive_hit = True
        if impulsive_hit:
            return "충동"

        return "중립"

    df["재해석"] = df.apply(classify, axis=1)
    return df

def month_days_from_str(month_str):
    y, m = month_str.split("-")
    y = int(y)
    m = int(m)
    return calendar.monthrange(y, m)[1]

def make_kr_currency(x):
    try:
        return f"{int(x):,}원"
    except Exception:
        return str(x)

def corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def line_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return None
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)

st.title("텅장 구조대")
st.caption("가계부 데이터를 기반으로 낭비 비용을 판별하고, 패턴 분석/경고/시뮬레이션을 제공합니다.")

with st.sidebar:
    st.header("데이터 입력")

    mode = st.radio(
        "데이터 불러오기",
        ["샘플 파일 사용", "CSV 업로드"],
        help="샘플 파일은 앱 폴더에 sample_household_ledger.csv, sample_monthly_budget.csv가 있어야 합니다."
    )

    ledger_file = None
    budget_file = None

    if mode == "CSV 업로드":
        ledger_file = st.file_uploader("가계부 원장 CSV 업로드", type=["csv"])
        budget_file = st.file_uploader("월급/예산 CSV 업로드", type=["csv"])
    else:
        ledger_file = str(Path(__file__).parent / DEFAULT_LEDGER)
        budget_file = str(Path(__file__).parent / DEFAULT_BUDGET)

    st.header("재해석 카테고리 판별 규칙")

    impulsive_threshold = st.number_input(
        "쇼핑 충동 판별 금액 기준(원)",
        min_value=10000,
        max_value=500000,
        value=80000,
        step=10000,
        help="쇼핑(카테고리=쇼핑)에서 이 금액 이상이면 '충동'으로 분류합니다."
    )

    neg_ratio_alert = st.slider(
        "충동+게으름 경고 기준(%)",
        min_value=10,
        max_value=80,
        value=30,
        step=5,
        help="전체 지출 중 (충동+게으름) 비중이 이 값 이상이면 경고 메시지를 띄웁니다."
    )

    breath_ratio_praise = st.slider(
        "호흡비용 칭찬 기준(%)",
        min_value=30,
        max_value=95,
        value=70,
        step=5,
        help="전체 지출 중 호흡비용 비중이 이 값 이상이면 '참 잘했어요' 메시지를 띄웁니다."
    )

    lazy_keywords_raw = st.text_input(
        "게으름 키워드(쉼표로 구분)",
        value="택시,대리,배달,지각,피곤",
        help="목적/비고에 포함되면 '게으름' 분류에 영향을 줍니다."
    )
    impulsive_keywords_raw = st.text_input(
        "충동 키워드(쉼표로 구분)",
        value="충동,특가,세일,무신사,쿠팡,쇼핑",
        help="목적/비고에 포함되면 '충동' 분류에 영향을 줍니다."
    )

    lazy_keywords = [x.strip() for x in lazy_keywords_raw.split(",") if x.strip()]
    impulsive_keywords = [x.strip() for x in impulsive_keywords_raw.split(",") if x.strip()]

    st.header("호들갑(예산 경고) 설정")

    asof_date_mode = st.radio(
        "기준일 선택",
        ["데이터 최신일 사용", "직접 선택"],
        help="이번달 예상 지출(호들갑)은 기준일까지의 지출 속도로 계산합니다. (현재까지 총 지출/경과일수)*한달일수"
    )

    sims = st.selectbox(
        "희망회로 시뮬레이션 기간(개월)",
        [3, 6, 9, 12, 18, 24],
        index=3,
        help="부정적 카테고리 절감 효과를 누적해서 계산할 기간입니다."
    )

    cut_pct = st.slider(
        "절감 비율(%)",
        min_value=0,
        max_value=80,
        value=10,
        step=5,
        help="충동+게으름 지출을 이 비율만큼 줄인다고 가정합니다."
    )

    travel_goal = st.number_input(
        "희망회로 목표 금액(원)",
        min_value=100000,
        max_value=20000000,
        value=2500000,
        step=100000,
        help="예: 여행자금/비상금 등 목표 금액입니다."
    )

try:
    ledger_df_raw = safe_read_csv(ledger_file)
    budget_df_raw = safe_read_csv(budget_file)

    if ledger_df_raw is None:
        raise ValueError("가계부 CSV를 불러오지 못했습니다.")
    if budget_df_raw is None:
        raise ValueError("월급/예산 CSV를 불러오지 못했습니다.")

    ledger_df = normalize_columns_ledger(ledger_df_raw)
    budget_df = normalize_columns_budget(budget_df_raw)

except Exception as e:
    st.error(str(e))
    st.stop()

if "ledger_edit" not in st.session_state:
    st.session_state["ledger_edit"] = ledger_df.copy()

if "budget_edit" not in st.session_state:
    st.session_state["budget_edit"] = budget_df.copy()

tabs = st.tabs(["입력/편집", "재해석 분석", "유혹 패턴", "텅장 방지 호들갑", "희망회로 시뮬레이터"])

with tabs[0]:
    st.subheader("가계부 데이터 입력/편집")
    st.caption("엑셀처럼 수정 가능합니다. 수정 후 아래 버튼으로 적용하세요. (제안서 Input: 날짜/시간/카테고리/목적/비용/비고):contentReference[oaicite:2]{index=2}")

    edited_ledger = st.data_editor(
        st.session_state["ledger_edit"],
        use_container_width=True,
        num_rows="dynamic"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("가계부 변경 적용"):
            st.session_state["ledger_edit"] = edited_ledger.copy()
            st.success("가계부 변경을 적용했습니다.")
    with c2:
        if st.button("가계부 원본으로 되돌리기"):
            st.session_state["ledger_edit"] = ledger_df.copy()
            st.success("가계부를 원본으로 되돌렸습니다.")

    st.subheader("월급/예산 데이터 입력/편집")
    edited_budget = st.data_editor(
        st.session_state["budget_edit"],
        use_container_width=True,
        num_rows="dynamic"
    )
    c3, c4 = st.columns(2)
    with c3:
        if st.button("예산 변경 적용"):
            st.session_state["budget_edit"] = edited_budget.copy()
            st.success("예산 변경을 적용했습니다.")
    with c4:
        if st.button("예산 원본으로 되돌리기"):
            st.session_state["budget_edit"] = budget_df.copy()
            st.success("예산을 원본으로 되돌렸습니다.")

with tabs[1]:
    st.subheader("재해석 카테고리 분석")
    st.caption("기존 카테고리를 '호흡/충동/게으름/성장'으로 재해석해 팩폭 피드백을 제공합니다.:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}")

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    budget_use = normalize_columns_budget(st.session_state["budget_edit"])

    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    m = st.selectbox("분석할 월 선택", sorted(classified["month"].unique()))
    one_month = classified[classified["month"] == m].copy()

    total_spend = float(one_month["비용"].sum())
    by_re = one_month.groupby("재해석")["비용"].sum().sort_values(ascending=False)
    neg_spend = float(by_re.get("충동", 0) + by_re.get("게으름", 0))
    breath_spend = float(by_re.get("호흡", 0))
    neg_ratio = 0.0 if total_spend == 0 else 100.0 * neg_spend / total_spend
    breath_ratio = 0.0 if total_spend == 0 else 100.0 * breath_spend / total_spend

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 월 총지출", make_kr_currency(total_spend))
    c2.metric("충동+게으름", make_kr_currency(neg_spend))
    c3.metric("부정비중", f"{neg_ratio:.1f}%")
    c4.metric("호흡비중", f"{breath_ratio:.1f}%")

    if neg_ratio >= float(neg_ratio_alert):
        st.error(f"소비 좀 줄이세요. (충동+게으름 비중 {neg_ratio:.1f}% ≥ {neg_ratio_alert}%)")
    if breath_ratio >= float(breath_ratio_praise):
        st.success(f"참 잘했어요! 고정비(호흡) 비중이 {breath_ratio:.1f}%로 높습니다.")

    pie_df = pd.DataFrame({"재해석": by_re.index, "비용": by_re.values})
    fig_pie = px.pie(pie_df, names="재해석", values="비용", title="재해석 카테고리별 지출 비중")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("부정적 카테고리 vs 월 총지출 상관관계")
    month_agg = classified.groupby("month").agg(
        total=("비용", "sum"),
        impulsive=("비용", lambda s: float(s[classified.loc[s.index, "재해석"].values == "충동"].sum())),
        lazy=("비용", lambda s: float(s[classified.loc[s.index, "재해석"].values == "게으름"].sum()))
    ).reset_index()

    month_agg["negative"] = month_agg["impulsive"] + month_agg["lazy"]
    corr = corr_pearson(month_agg["negative"].values, month_agg["total"].values)

    c5, c6 = st.columns(2)
    with c5:
        st.metric("피어슨 상관계수(부정 vs 총지출)", "-" if np.isnan(corr) else f"{corr:.3f}")
        st.caption("값이 1에 가까울수록 '부정 지출이 커질수록 총지출도 커지는 경향'이 강합니다.")
    with c6:
        fig_sc = px.scatter(
            month_agg,
            x="negative",
            y="total",
            text="month",
            title="월별 부정 지출(충동+게으름) vs 월 총지출"
        )
        fit = line_fit(month_agg["negative"].values, month_agg["total"].values)
        if fit is not None:
            a, b = fit
            xs = np.linspace(month_agg["negative"].min(), month_agg["negative"].max(), 50)
            ys = a * xs + b
            fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
        fig_sc.update_traces(textposition="top center")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("재해석된 데이터 테이블")
    st.dataframe(one_month[["날짜", "시간", "카테고리", "목적", "비용", "비고", "재해석"]], use_container_width=True)

    dl = one_month.to_csv(index=False).encode("utf-8-sig")
    st.download_button("선택 월 재해석 결과 CSV 다운로드", data=dl, file_name=f"reinterpreted_{m}.csv", mime="text/csv")

with tabs[2]:
    st.subheader("유혹 패턴 분석")
    st.caption("충동/게으름 비용이 많이 발생하는 요일과 시간대를 추출해 시각화합니다.:contentReference[oaicite:5]{index=5}")

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    only_neg = classified[classified["재해석"].isin(["충동", "게으름"])].copy()
    if len(only_neg) == 0:
        st.warning("충동/게으름으로 분류된 지출이 없습니다. 키워드/기준금액을 조정해보세요.")
    else:
        heat = (
            only_neg.groupby(["weekday_name", "hour"])["비용"]
            .sum()
            .reset_index()
        )
        order_week = ["월", "화", "수", "목", "금", "토", "일"]
        heat["weekday_name"] = pd.Categorical(heat["weekday_name"], categories=order_week, ordered=True)
        pivot = heat.pivot_table(index="weekday_name", columns="hour", values="비용", fill_value=0).reindex(order_week)

        fig_hm = px.imshow(
            pivot.values,
            x=pivot.columns.astype(int),
            y=pivot.index.astype(str),
            aspect="auto",
            title="충동/게으름 지출 히트맵 (요일 × 시간)"
        )
        fig_hm.update_layout(xaxis_title="시간(시)", yaxis_title="요일")
        st.plotly_chart(fig_hm, use_container_width=True)

        top = only_neg.groupby(["weekday_name", "hour", "재해석"])["비용"].sum().reset_index().sort_values("비용", ascending=False).head(10)
        st.subheader("가장 취약한 TOP 10 (요일/시간/유형)")
        top["비용"] = top["비용"].apply(make_kr_currency)
        st.dataframe(top, use_container_width=True)

with tabs[3]:
    st.subheader("텅장 방지 호들갑 기능")
    st.caption("현재까지의 지출 속도로 이번달 예상 총지출을 계산하고, 예산 대비 초과 여부를 경고합니다.:contentReference[oaicite:6]{index=6}")

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    budget_use = normalize_columns_budget(st.session_state["budget_edit"])

    months = sorted(ledger_use["month"].unique())
    m = st.selectbox("호들갑을 볼 월 선택", months, index=len(months)-1)

    y, mo = map(int, m.split("-"))
    last_day = month_days_from_str(m)
    month_start = pd.Timestamp(y, mo, 1)
    month_end = pd.Timestamp(y, mo, last_day)

    month_df = ledger_use[(ledger_use["날짜"] >= month_start) & (ledger_use["날짜"] <= month_end)].copy()
    if len(month_df) == 0:
        st.warning("선택 월 데이터가 없습니다.")
    else:
        max_date = month_df["날짜"].max()

        if asof_date_mode == "직접 선택":
            asof = st.date_input("기준일(이 날까지 지출을 반영)", value=max_date.date(), min_value=month_start.date(), max_value=month_end.date())
            asof = pd.Timestamp(asof)
        else:
            asof = pd.Timestamp(max_date.date())

        spent_to_date = float(month_df[month_df["날짜"] <= asof]["비용"].sum())
        elapsed_days = int((asof - month_start).days) + 1
        days_in_month = int(last_day)

        if elapsed_days <= 0:
            st.error("기준일이 월 시작보다 이전입니다.")
        else:
            forecast_total = (spent_to_date / elapsed_days) * days_in_month

            b_row = budget_use[budget_use["month"] == m]
            if len(b_row) == 0:
                budget_amount = 0
                salary = 0
                payday = 25
            else:
                budget_amount = int(b_row.iloc[0]["budget"])
                salary = int(b_row.iloc[0]["salary"])
                payday = int(b_row.iloc[0]["payday"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재까지 지출", make_kr_currency(spent_to_date))
            c2.metric("경과일수", f"{elapsed_days}일 / {days_in_month}일")
            c3.metric("이번달 예상 총지출", make_kr_currency(forecast_total))
            c4.metric("월 예산", make_kr_currency(budget_amount))

            comp_df = pd.DataFrame(
                {"항목": ["예산", "예상 지출"], "금액": [budget_amount, forecast_total]}
            )
            fig_bar = px.bar(comp_df, x="항목", y="금액", title="예산 대비 예상 지출")
            st.plotly_chart(fig_bar, use_container_width=True)

            avg_daily = spent_to_date / max(elapsed_days, 1)
            remaining = float(budget_amount) - float(spent_to_date)
            if avg_daily <= 0:
                st.info("현재까지 지출이 거의 없어 예측이 불안정합니다.")
            else:
                if remaining <= 0:
                    st.error("이미 예산을 초과했습니다. 텅장 임박.")
                else:
                    days_left_budget = int(np.floor(remaining / avg_daily))
                    exhaust_day = asof + pd.Timedelta(days=days_left_budget)
                    pay_date = pd.Timestamp(y, mo, min(payday, last_day))

                    if exhaust_day < pay_date:
                        diff = int((pay_date - exhaust_day).days)
                        st.error(f"현재 템포로 소비하면 월급날 {diff}일 전에 예산이 소진됩니다.")
                    else:
                        st.success("현재 템포로는 월급날 전까지 예산이 유지될 가능성이 큽니다.")

            classified = assign_reinterpreted_category(
                normalize_columns_ledger(st.session_state["ledger_edit"]),
                impulsive_threshold=int(impulsive_threshold),
                lazy_keywords=lazy_keywords,
                impulsive_keywords=impulsive_keywords
            )
            month_class = classified[classified["month"] == m]
            total = float(month_class["비용"].sum())
            imp = float(month_class[month_class["재해석"] == "충동"]["비용"].sum())
            lazy = float(month_class[month_class["재해석"] == "게으름"]["비용"].sum())

            saved_if_cut = 0.10 * (imp + lazy)
            st.info(f"충동+게으름 비용을 10%만 줄여도 약 {make_kr_currency(saved_if_cut)} 절약 가능합니다.")

with tabs[4]:
    st.subheader("희망회로 시뮬레이터")
    st.caption("부정적 지출(충동/게으름)을 절감했을 때 누적 절감액과 목표 달성 가능성을 보여줍니다.:contentReference[oaicite:7]{index=7}")

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    month_sum = classified.groupby(["month", "재해석"])["비용"].sum().reset_index()
    pivot = month_sum.pivot_table(index="month", columns="재해석", values="비용", fill_value=0).reset_index()
    if "충동" not in pivot.columns:
        pivot["충동"] = 0
    if "게으름" not in pivot.columns:
        pivot["게으름"] = 0

    pivot["부정합"] = pivot["충동"] + pivot["게으름"]
    avg_neg = float(pivot["부정합"].mean()) if len(pivot) > 0 else 0.0

    saved = avg_neg * (float(cut_pct) / 100.0) * int(sims)
    progress = 0.0 if float(travel_goal) <= 0 else min(1.0, saved / float(travel_goal))

    c1, c2, c3 = st.columns(3)
    c1.metric("월 평균 부정 지출(충동+게으름)", make_kr_currency(avg_neg))
    c2.metric(f"{cut_pct}% 절감 × {sims}개월", make_kr_currency(saved))
    c3.metric("목표 대비 달성률", f"{progress*100:.1f}%")

    st.progress(progress)

    if saved >= float(travel_goal):
        st.success("목표 달성 가능! 희망회로가 아니라 현실회로입니다.")
    else:
        gap = float(travel_goal) - saved
        st.warning(f"아직 {make_kr_currency(gap)} 부족합니다. 절감 비율/기간을 조정해보세요.")

    fig_trend = px.line(
        pivot,
        x="month",
        y="부정합",
        title="월별 부정 지출(충동+게으름) 추이"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("희망회로 시각화(예: 서울 → LA)")
    st.caption("제안서 예시처럼 지도/경로를 간단히 시각화합니다. (실제 여행비는 사용자가 목표 금액으로 정의)")

    dest = st.selectbox("목표 여행지(예시)", ["Los Angeles", "Tokyo", "Bangkok", "Singapore", "Sydney"], index=0)

    coords = {
        "Seoul": (37.5665, 126.9780),
        "Los Angeles": (34.0522, -118.2437),
        "Tokyo": (35.6762, 139.6503),
        "Bangkok": (13.7563, 100.5018),
        "Singapore": (1.3521, 103.8198),
        "Sydney": (-33.8688, 151.2093),
    }

    o_lat, o_lon = coords["Seoul"]
    d_lat, d_lon = coords[dest]

    fig_geo = go.Figure()
    fig_geo.add_trace(
        go.Scattergeo(
            lon=[o_lon, d_lon],
            lat=[o_lat, d_lat],
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
            name="Route"
        )
    )
    fig_geo.update_layout(
        title=f"Seoul → {dest}",
        geo=dict(
            projection_type="natural earth",
            showland=True
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_geo, use_container_width=True)
