import re
import calendar
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="텅장 구조대", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LEDGER = "sample_household_ledger.csv"
DEFAULT_BUDGET = "sample_monthly_budget.csv"
DEFAULT_GOODPRICE = "서울시 착한가격업소 현황.csv"

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

def make_kr_currency(x):
    try:
        return f"{int(round(float(x))):,}원"
    except Exception:
        return str(x)

def month_days_from_str(month_str):
    y, m = month_str.split("-")
    y = int(y)
    m = int(m)
    return calendar.monthrange(y, m)[1]

def normalize_columns_ledger(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
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
    df["비용"] = df["비용"].astype(str).str.replace(",", "", regex=False).str.strip()
    df["비용"] = pd.to_numeric(df["비용"], errors="coerce").fillna(0).astype(int)

    def parse_hour(x):
        s = str(x)
        if ":" in s:
            try:
                return int(s.split(":")[0])
            except Exception:
                return np.nan
        return np.nan

    df["hour"] = df["시간"].apply(parse_hour).fillna(12).astype(int)
    df["weekday"] = df["날짜"].dt.weekday
    df["weekday_name"] = df["weekday"].map({0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"})
    df["month"] = df["날짜"].dt.to_period("M").astype(str)
    df = df.sort_values(["날짜", "시간"]).reset_index(drop=True)
    return df

def normalize_columns_budget(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["month", "salary", "budget", "payday"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"월급/예산 CSV에 필요한 컬럼이 없습니다: {missing}")

    df["month"] = df["month"].astype(str).str.strip()
    for c in ["salary", "budget", "payday"]:
        df[c] = df[c].astype(str).str.replace(",", "", regex=False).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["salary"] = df["salary"].fillna(0).astype(int)
    df["budget"] = df["budget"].fillna(0).astype(int)
    df["payday"] = df["payday"].fillna(25).astype(int)
    df = df.sort_values("month").reset_index(drop=True)
    return df

def extract_gu(addr):
    s = str(addr)
    m = re.search(r"([가-힣]+구)", s)
    if m:
        return m.group(1)
    return ""

def normalize_goodprice(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    def pick_col(candidates):
        for cand in candidates:
            for c in df.columns:
                if cand == c:
                    return c
        for cand in candidates:
            for c in df.columns:
                if cand in c:
                    return c
        return None

    c_name = pick_col(["업소명"])
    c_ind = pick_col(["분류코드명", "업종", "업태", "분류"])
    c_addr = pick_col(["업소 주소", "주소", "도로명주소", "지번주소"])
    c_route = pick_col(["찾아오시는 길", "찾아오시는길", "찾아"])
    c_info = pick_col(["업소정보", "업소 정보"])
    c_pride = pick_col(["자랑거리", "자랑"])
    c_like = pick_col(["추천수", "추천 수", "추천"])
    c_photo = pick_col(["업소 사진", "사진", "이미지", "photo", "Photo"])

    if c_name is None:
        df["업소명"] = ""
    else:
        df["업소명"] = df[c_name].astype(str).fillna("").str.strip()

    if c_ind is None:
        df["업종"] = ""
    else:
        df["업종"] = df[c_ind].astype(str).fillna("").str.strip()

    if c_addr is None:
        df["주소"] = ""
    else:
        df["주소"] = df[c_addr].astype(str).fillna("").str.strip()

    df["찾아오시는길"] = df[c_route].astype(str).fillna("").str.strip() if c_route is not None else ""
    df["업소정보"] = df[c_info].astype(str).fillna("").str.strip() if c_info is not None else ""
    df["자랑거리"] = df[c_pride].astype(str).fillna("").str.strip() if c_pride is not None else ""

    if c_like is None:
        df["추천수"] = 0
    else:
        s = df[c_like].astype(str).str.replace(",", "", regex=False).str.strip()
        df["추천수"] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    df["사진"] = df[c_photo].astype(str).fillna("").str.strip() if c_photo is not None else ""

    df["구"] = df["주소"].apply(extract_gu)
    df["text"] = (df["업소명"] + " " + df["업종"] + " " + df["주소"] + " " + df["찾아오시는길"] + " " + df["업소정보"] + " " + df["자랑거리"]).str.strip()
    df = df.reset_index(drop=True)
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

def build_intent_dict():
    return {
        "혼밥/점심": ["혼밥", "점심", "백반", "정식", "1인", "가성비", "런치", "식사", "회사", "근처"],
        "모임/회식": ["회식", "모임", "단체", "예약", "술", "친구", "약속"],
        "배달대체": ["배달", "배달앱", "야근", "간편", "포장", "테이크아웃", "빠른"],
        "접근성": ["역", "정류장", "시장", "도보", "가까", "근처", "출구", "사거리"],
        "미용": ["미용", "커트", "염색", "펌", "헤어", "이발", "이미용"],
        "세탁": ["세탁", "드라이", "수선", "크리닝", "세탁소"],
        "가성비": ["저렴", "착한가격", "가성비", "할인", "저가", "합리"],
    }

def build_user_text(df_month):
    t = (df_month["카테고리"].astype(str) + " " + df_month["목적"].astype(str) + " " + df_month["비고"].astype(str)).str.cat(sep=" ")
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t

def build_user_intents(user_text, intent_dict):
    u = str(user_text)
    d = {}
    for intent, kws in intent_dict.items():
        cnt = 0
        for k in kws:
            if k in u:
                cnt += 1
        d[intent] = cnt
    return d

def score_rule_based(store_text, user_intents, intent_dict, intent_weight):
    t = str(store_text)
    s = 0.0
    reasons = []
    for intent, kws in intent_dict.items():
        ucnt = int(user_intents.get(intent, 0))
        if ucnt <= 0:
            continue
        hit = 0
        for k in kws:
            if k in t:
                hit += 1
        if hit > 0:
            add = float(intent_weight.get(intent, 1.0)) * float(min(3, hit)) * float(min(3, ucnt))
            s += add
            reasons.append(f"{intent}({hit})")
    return s, ", ".join(reasons)

def map_ledger_to_store_industries(df_month):
    text = build_user_text(df_month)

    want_food = any(k in text for k in ["식사", "점심", "저녁", "배달", "회식", "카페", "간식"])
    want_beauty = any(k in text for k in ["미용", "커트", "염색", "펌", "이미용"])
    want_laundry = any(k in text for k in ["세탁", "드라이", "수선", "크리닝"])

    industries = set()
    if want_food:
        industries |= set(["한식", "중식", "기타외식업", "경양식/일식", "일식", "경양식"])
    if want_beauty:
        industries |= set(["이미용"])
    if want_laundry:
        industries |= set(["세탁", "세탁업"])
    if len(industries) == 0:
        industries |= set(["한식", "중식", "기타외식업", "이미용", "세탁", "기타서비스업"])
    return industries

st.title("텅장 구조대")
st.caption("가계부 기반 지출 진단 + 착한가격업소 추천(룰 기반 only)")

with st.sidebar:
    st.header("데이터 입력")
    mode = st.radio("데이터 불러오기", ["프로젝트 폴더 파일 사용", "CSV 업로드"])

    if mode == "CSV 업로드":
        ledger_file = st.file_uploader("가계부 원장 CSV", type=["csv"])
        budget_file = st.file_uploader("월급/예산 CSV", type=["csv"])
        goodprice_file = st.file_uploader("착한가격업소 CSV", type=["csv"])
    else:
        ledger_file = str(PROJECT_ROOT / DEFAULT_LEDGER)
        budget_file = str(PROJECT_ROOT / DEFAULT_BUDGET)
        goodprice_file = str(PROJECT_ROOT / DEFAULT_GOODPRICE)

    st.header("재해석 분류 규칙")
    impulsive_threshold = st.number_input("쇼핑 충동 판별 금액(원)", min_value=10000, max_value=500000, value=80000, step=10000)
    neg_ratio_alert = st.slider("충동+게으름 경고 기준(%)", min_value=10, max_value=80, value=30, step=5)
    breath_ratio_praise = st.slider("호흡비용 칭찬 기준(%)", min_value=30, max_value=95, value=70, step=5)

    lazy_keywords_raw = st.text_input("게으름 키워드(쉼표)", value="택시,대리,배달,지각,피곤")
    impulsive_keywords_raw = st.text_input("충동 키워드(쉼표)", value="충동,특가,세일,무신사,쿠팡,쇼핑")
    lazy_keywords = [x.strip() for x in lazy_keywords_raw.split(",") if x.strip()]
    impulsive_keywords = [x.strip() for x in impulsive_keywords_raw.split(",") if x.strip()]

    st.header("호들갑/절감 설정")
    asof_date_mode = st.radio("기준일", ["데이터 최신일 사용", "직접 선택"])
    sims = st.selectbox("절감 누적 기간(개월)", [3, 6, 9, 12, 18, 24], index=3)
    cut_pct = st.slider("충동+게으름 절감 비율(%)", min_value=0, max_value=80, value=10, step=5)
    save_goal = st.number_input("절약 목표 금액(원)", min_value=100000, max_value=20000000, value=2500000, step=100000)

    st.header("착한가격 추천")
    run_reco = st.button("착한가격 추천 갱신")

try:
    ledger_df_raw = safe_read_csv(ledger_file)
    budget_df_raw = safe_read_csv(budget_file)
    good_df_raw = safe_read_csv(goodprice_file)

    if ledger_df_raw is None:
        raise ValueError("가계부 CSV를 불러오지 못했습니다.")
    if budget_df_raw is None:
        raise ValueError("월급/예산 CSV를 불러오지 못했습니다.")
    if good_df_raw is None:
        raise ValueError("착한가격업소 CSV를 불러오지 못했습니다.")

    ledger_df = normalize_columns_ledger(ledger_df_raw)
    budget_df = normalize_columns_budget(budget_df_raw)
    good_df = normalize_goodprice(good_df_raw)

except Exception as e:
    st.error(str(e))
    st.stop()

if "ledger_edit" not in st.session_state:
    st.session_state["ledger_edit"] = ledger_df.copy()

if "budget_edit" not in st.session_state:
    st.session_state["budget_edit"] = budget_df.copy()

tabs = st.tabs(["입력/편집", "재해석 분석", "유혹 패턴", "텅장 방지 호들갑", "절약 액션: 착한가격 추천"])

with tabs[0]:
    st.subheader("가계부 데이터 입력/편집")
    edited_ledger = st.data_editor(st.session_state["ledger_edit"], use_container_width=True, num_rows="dynamic")
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
    edited_budget = st.data_editor(st.session_state["budget_edit"], use_container_width=True, num_rows="dynamic")
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

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    months = sorted(classified["month"].unique())
    m = st.selectbox("분석할 월", months, index=max(0, len(months) - 1))
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
        st.error(f"충동+게으름 비중 {neg_ratio:.1f}% (기준 {neg_ratio_alert}%)")
    if breath_ratio >= float(breath_ratio_praise):
        st.success(f"호흡비중 {breath_ratio:.1f}% (기준 {breath_ratio_praise}%)")

    pie_df = pd.DataFrame({"재해석": by_re.index, "비용": by_re.values})
    fig_pie = px.pie(pie_df, names="재해석", values="비용", title="재해석 카테고리별 지출 비중")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("부정 지출(충동+게으름)과 총지출의 월별 관계")
    month_agg = classified.groupby("month").agg(total=("비용", "sum")).reset_index()
    month_agg["negative"] = classified[classified["재해석"].isin(["충동", "게으름"])].groupby("month")["비용"].sum().reindex(month_agg["month"]).fillna(0).values
    corr = corr_pearson(month_agg["negative"].values, month_agg["total"].values)

    c5, c6 = st.columns(2)
    with c5:
        st.metric("피어슨 상관(부정 vs 총지출)", "-" if np.isnan(corr) else f"{corr:.3f}")
        st.caption("값이 클수록 부정 지출이 커질 때 총지출도 같이 커지는 경향")
    with c6:
        fig_sc = px.scatter(month_agg, x="negative", y="total", text="month", title="월별 부정지출 vs 총지출")
        fit = line_fit(month_agg["negative"].values, month_agg["total"].values)
        if fit is not None:
            a, b = fit
            xs = np.linspace(month_agg["negative"].min(), month_agg["negative"].max(), 50)
            ys = a * xs + b
            fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="추세선"))
        fig_sc.update_traces(textposition="top center")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("재해석 데이터")
    st.dataframe(one_month[["날짜", "시간", "카테고리", "목적", "비용", "비고", "재해석"]], use_container_width=True)

with tabs[2]:
    st.subheader("유혹 패턴 분석(요일 × 시간)")
    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    only_neg = classified[classified["재해석"].isin(["충동", "게으름"])].copy()
    if len(only_neg) == 0:
        st.warning("충동/게으름 지출이 없습니다. 기준(키워드/금액)을 조정해보세요.")
    else:
        heat = only_neg.groupby(["weekday_name", "hour"])["비용"].sum().reset_index()
        order_week = ["월", "화", "수", "목", "금", "토", "일"]
        heat["weekday_name"] = pd.Categorical(heat["weekday_name"], categories=order_week, ordered=True)
        pivot = heat.pivot_table(index="weekday_name", columns="hour", values="비용", fill_value=0).reindex(order_week)

        fig_hm = px.imshow(
            pivot.values,
            x=pivot.columns.astype(int),
            y=pivot.index.astype(str),
            aspect="auto",
            title="충동/게으름 지출 히트맵"
        )
        fig_hm.update_layout(xaxis_title="시간(시)", yaxis_title="요일")
        st.plotly_chart(fig_hm, use_container_width=True)

        top = only_neg.groupby(["weekday_name", "hour", "재해석"])["비용"].sum().reset_index().sort_values("비용", ascending=False).head(12)
        top_disp = top.copy()
        top_disp["비용"] = top_disp["비용"].apply(make_kr_currency)
        st.subheader("취약 TOP 12")
        st.dataframe(top_disp, use_container_width=True)

with tabs[3]:
    st.subheader("텅장 방지 호들갑")
    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    budget_use = normalize_columns_budget(st.session_state["budget_edit"])

    months = sorted(ledger_use["month"].unique())
    m = st.selectbox("호들갑을 볼 월", months, index=max(0, len(months) - 1))

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
            asof = st.date_input("기준일", value=max_date.date(), min_value=month_start.date(), max_value=month_end.date())
            asof = pd.Timestamp(asof)
        else:
            asof = pd.Timestamp(max_date.date())

        spent_to_date = float(month_df[month_df["날짜"] <= asof]["비용"].sum())
        elapsed_days = int((asof - month_start).days) + 1
        days_in_month = int(last_day)
        forecast_total = (spent_to_date / max(1, elapsed_days)) * days_in_month

        b_row = budget_use[budget_use["month"] == m]
        if len(b_row) == 0:
            budget_amount = 0
            payday = 25
        else:
            budget_amount = int(b_row.iloc[0]["budget"])
            payday = int(b_row.iloc[0]["payday"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재까지 지출", make_kr_currency(spent_to_date))
        c2.metric("경과일수", f"{elapsed_days}일 / {days_in_month}일")
        c3.metric("이번달 예상 총지출", make_kr_currency(forecast_total))
        c4.metric("월 예산", make_kr_currency(budget_amount))

        comp_df = pd.DataFrame({"항목": ["예산", "예상 지출"], "금액": [budget_amount, forecast_total]})
        fig_bar = px.bar(comp_df, x="항목", y="금액", title="예산 대비 예상 지출")
        st.plotly_chart(fig_bar, use_container_width=True)

        avg_daily = spent_to_date / max(1, elapsed_days)
        remaining = float(budget_amount) - float(spent_to_date)
        pay_date = pd.Timestamp(y, mo, min(payday, last_day))

        if budget_amount <= 0:
            st.warning("선택 월의 예산 데이터가 없습니다. 예산 CSV를 확인하세요.")
        else:
            if remaining <= 0:
                st.error("이미 예산을 초과했습니다. 텅장 임박.")
            else:
                days_left_budget = int(np.floor(remaining / max(1.0, avg_daily)))
                exhaust_day = asof + pd.Timedelta(days=days_left_budget)
                if exhaust_day < pay_date:
                    diff = int((pay_date - exhaust_day).days)
                    st.error(f"현재 템포면 월급날 {diff}일 전에 예산이 소진될 가능성이 큽니다.")
                else:
                    st.success("현재 템포로는 월급날 전까지 예산이 유지될 가능성이 큽니다.")

        classified = assign_reinterpreted_category(
            ledger_use,
            impulsive_threshold=int(impulsive_threshold),
            lazy_keywords=lazy_keywords,
            impulsive_keywords=impulsive_keywords
        )
        month_class = classified[classified["month"] == m]
        imp = float(month_class[month_class["재해석"] == "충동"]["비용"].sum())
        lazy = float(month_class[month_class["재해석"] == "게으름"]["비용"].sum())
        saved_if_cut = (float(cut_pct) / 100.0) * (imp + lazy)
        st.info(f"충동+게으름 비용을 {cut_pct}% 줄이면, 이번 달 기준 약 {make_kr_currency(saved_if_cut)} 절약 가능")

with tabs[4]:
    st.subheader("절약 액션: 착한가격업소 추천(룰 기반 only)")
    st.caption("사용자(충동/게으름 지출) 텍스트 패턴과 업소 텍스트를 키워드 사전(dict)로 매칭해 추천합니다.")

    ledger_use = normalize_columns_ledger(st.session_state["ledger_edit"])
    classified = assign_reinterpreted_category(
        ledger_use,
        impulsive_threshold=int(impulsive_threshold),
        lazy_keywords=lazy_keywords,
        impulsive_keywords=impulsive_keywords
    )

    months = sorted(classified["month"].unique())
    m = st.selectbox("추천에 사용할 월(패턴 기준)", months, index=max(0, len(months) - 1))
    df_m = classified[classified["month"] == m].copy()

    neg_df = df_m[df_m["재해석"].isin(["충동", "게으름"])].copy()
    if len(neg_df) == 0:
        st.warning("선택 월에 충동/게으름 지출이 없습니다. 다른 월을 선택하거나 기준을 조정하세요.")
        st.stop()

    total_spend = float(df_m["비용"].sum())
    neg_spend = float(neg_df["비용"].sum())
    neg_ratio = 0.0 if total_spend == 0 else 100.0 * neg_spend / total_spend

    c1, c2, c3 = st.columns(3)
    c1.metric("선택 월 총지출", make_kr_currency(total_spend))
    c2.metric("충동+게으름", make_kr_currency(neg_spend))
    c3.metric("부정비중", f"{neg_ratio:.1f}%")

    intent_dict = build_intent_dict()
    user_text = build_user_text(neg_df)
    user_intents = build_user_intents(user_text, intent_dict)

    intent_weight = {k: 1.0 for k in intent_dict.keys()}
    intent_weight["접근성"] = 1.2
    intent_weight["배달대체"] = 1.2
    intent_weight["가성비"] = 1.1

    st.markdown("### 1) 생활권 선택")
    all_gu = sorted([g for g in good_df["구"].unique().tolist() if g])
    gu = st.selectbox("거주/활동 구(주소 기준)", ["(선택안함)"] + all_gu, index=0)

    st.markdown("### 2) 업종 자동 추천(패턴 기반) + 수동 조정")
    suggested_ind = map_ledger_to_store_industries(neg_df)
    ind_all = sorted([x for x in good_df["업종"].unique().tolist() if str(x).strip() and str(x).strip().lower() != "nan"])
    ind_default = [x for x in ind_all if any(s in x for s in suggested_ind)]
    chosen_ind = st.multiselect("추천할 업종 선택", options=ind_all, default=ind_default[:6] if len(ind_default) > 0 else ind_all[:6])

    st.markdown("### 3) 절감 목표(희망회로)")
    avg_neg_month = float(classified[classified["재해석"].isin(["충동", "게으름"])].groupby("month")["비용"].sum().mean())
    saved = avg_neg_month * (float(cut_pct) / 100.0) * int(sims)
    progress = 0.0 if float(save_goal) <= 0 else min(1.0, saved / float(save_goal))

    c4, c5, c6 = st.columns(3)
    c4.metric("월 평균 부정 지출", make_kr_currency(avg_neg_month))
    c5.metric(f"{cut_pct}% 절감 × {sims}개월", make_kr_currency(saved))
    c6.metric("목표 대비 달성률", f"{progress * 100:.1f}%")
    st.progress(progress)

    st.markdown("### 4) 추천 결과")
    if not run_reco:
        st.caption("왼쪽 사이드바의 '착한가격 추천 갱신' 버튼을 눌러 추천을 생성하세요.")
        st.stop()

    cand = good_df.copy()
    if gu != "(선택안함)":
        cand = cand[cand["구"] == gu].copy()
    if chosen_ind:
        cand = cand[cand["업종"].isin(chosen_ind)].copy()

    if len(cand) == 0:
        st.warning("선택한 생활권/업종 조건에서 후보가 없습니다. 필터를 완화해보세요.")
        st.stop()

    pop = cand["추천수"].astype(float)
    pop_norm = (pop - pop.min()) / (pop.max() - pop.min()) if pop.max() != pop.min() else pop * 0.0

    rule_scores = []
    rule_reasons = []
    for t in cand["text"].tolist():
        s, r = score_rule_based(t, user_intents, intent_dict, intent_weight)
        rule_scores.append(s)
        rule_reasons.append(r)

    rule_scores = np.asarray(rule_scores, dtype=float)
    rule_norm = (rule_scores - np.min(rule_scores)) / (np.max(rule_scores) - np.min(rule_scores)) if np.max(rule_scores) != np.min(rule_scores) else rule_scores * 0.0

    w_pop = 0.45
    w_rule = 0.55
    final = w_pop * pop_norm + w_rule * rule_norm

    out = cand[["업소명", "업종", "구", "주소", "찾아오시는길", "업소정보", "자랑거리", "추천수", "사진"]].copy()
    out["룰적합도"] = rule_scores
    out["룰근거"] = rule_reasons
    out["추천점수"] = final
    out = out.sort_values("추천점수", ascending=False).reset_index(drop=True)

    topn = st.slider("추천 표시 개수", min_value=5, max_value=30, value=10, step=5)
    show = out.head(int(topn)).copy()

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("#### TOP 추천 리스트")
        cols = ["업소명", "업종", "구", "주소", "추천수", "추천점수", "룰근거"]
        st.dataframe(show[cols], use_container_width=True)

    with cB:
        st.markdown("#### 추천 점수 구성(요약)")
        comp = pd.DataFrame({"구성요소": ["추천수(정규화)", "룰 기반(정규화)"], "가중치": [w_pop, w_rule]})
        fig = px.bar(comp, x="구성요소", y="가중치", title="추천점수 가중치")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 업소 카드(상세)")
    for i in range(min(len(show), 8)):
        r = show.iloc[i]
        with st.expander(f"{i+1}. {r['업소명']} | {r['업종']} | {r['구']} | 추천수 {int(r['추천수'])}", expanded=False):
            st.write(f"주소: {r['주소']}")
            if str(r["찾아오시는길"]).strip():
                st.write(f"찾아오시는 길: {r['찾아오시는길']}")
            if str(r["업소정보"]).strip():
                st.write(f"업소정보: {r['업소정보']}")
            if str(r["자랑거리"]).strip():
                st.write(f"자랑거리: {r['자랑거리']}")
            st.write(f"룰 적합도 근거: {r['룰근거'] if str(r['룰근거']).strip() else '매칭 근거가 적어 기본 랭킹(추천수/필터) 중심'}")
            st.write(f"추천점수: {float(r['추천점수']):.3f}")
            if str(r["사진"]).strip() and str(r["사진"]).lower().startswith("http"):
                st.markdown(f"사진 링크: {r['사진']}")

    dl = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button("추천 결과 CSV 다운로드", data=dl, file_name=f"goodprice_reco_{m}.csv", mime="text/csv")
