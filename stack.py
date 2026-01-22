import math
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="적금 목표 달성 확률 시뮬레이터", layout="wide")

DATA_FILE = "가계의 목적별 최종소비지출(계절조정, 명목, 분기)_21191430.csv"

@st.cache_data(show_spinner=False)
def load_local_quarterly_csv():
    p = Path(__file__).parent / DATA_FILE
    if not p.exists():
        raise FileNotFoundError(f"프로젝트 폴더에 데이터 파일이 없습니다: {p.name}")

    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    if len(df.columns) < 2:
        raise ValueError("CSV 컬럼이 2개 이상이어야 합니다. (분기/값)")

    date_col = df.columns[0]
    value_col = df.columns[1]

    d = df[[date_col, value_col]].copy()
    d.columns = ["date_raw", "value_raw"]

    d["date_raw"] = d["date_raw"].astype(str).str.strip()
    d["value_raw"] = d["value_raw"].astype(str).str.replace(",", "", regex=False).str.strip()
    d["value"] = pd.to_numeric(d["value_raw"], errors="coerce")
    d = d.dropna(subset=["value"]).reset_index(drop=True)

    def to_quarter_period(x):
        s = str(x).strip()
        s = s.replace("/Q", "Q")
        s = s.replace("-", "")
        try:
            return pd.Period(s, freq="Q")
        except Exception:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                raise ValueError(f"분기 파싱 실패: {x}")
            return pd.Period(dt, freq="Q")

    d["q"] = d["date_raw"].apply(to_quarter_period)
    d = d.sort_values("q").drop_duplicates("q").set_index("q")
    s = d["value"].astype(float).rename("consumption")
    return s

def build_stress_from_consumption(consumption, vol_window):
    c = consumption.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(c) < vol_window + 2:
        raise ValueError("데이터 길이가 짧아 변동성(rolling std)을 계산하기 어렵습니다. window를 줄이거나 데이터 기간을 늘려주세요.")
    g = np.log(c).diff()
    v = g.rolling(int(vol_window)).std()
    s = v.dropna().rename("stress")
    return s

def normalize_01(s):
    s = pd.Series(s).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.ffill().bfill().fillna(0.0)
    if len(s) < 2:
        return s
    mn = float(s.min())
    mx = float(s.max())
    if mx - mn == 0:
        return s * 0.0
    return (s - mn) / (mx - mn)

def rolling_forecast_rolling_mean(stress, k, min_train):
    s = stress.dropna().astype(float)
    idx = s.index
    y = s.values
    out = pd.Series(index=idx, dtype=float)
    k = int(max(1, k))
    min_train = int(max(2, min_train))
    for i in range(len(y) - 1):
        t = i + 1
        if t < min_train:
            continue
        start = max(0, t - k)
        pred = float(np.mean(y[start:t]))
        out.iloc[i + 1] = pred
    return out

def fit_ar1(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0, 1.0
    x = y[:-1]
    yy = y[1:]
    X = np.column_stack([np.ones_like(x), x])
    beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    a = float(beta[0])
    b = float(beta[1])
    return a, b

def rolling_forecast_ar1(stress, min_train):
    s = stress.dropna().astype(float)
    idx = s.index
    y = s.values
    out = pd.Series(index=idx, dtype=float)
    min_train = int(max(2, min_train))
    for i in range(len(y) - 1):
        t = i + 1
        if t < min_train:
            continue
        a, b = fit_ar1(y[:t])
        pred = a + b * y[t - 1]
        out.iloc[i + 1] = float(pred)
    return out

def forecast_future_quarters(stress, model_name, horizon_q, rm_k):
    s = stress.dropna().astype(float)
    horizon_q = int(max(1, horizon_q))
    rm_k = int(max(1, rm_k))

    if len(s) == 0:
        last_q = pd.Period("2020Q1", freq="Q")
        future_idx = pd.period_range(last_q + 1, last_q + horizon_q, freq="Q")
        return pd.Series(index=future_idx, data=[0.0] * horizon_q, dtype=float)

    last_q = s.index[-1]
    future_idx = pd.period_range(last_q + 1, last_q + horizon_q, freq="Q")

    if model_name == "Rolling Mean":
        y = s.values.copy()
        preds = []
        for _ in range(horizon_q):
            k = min(rm_k, len(y))
            pred = float(np.mean(y[-k:]))
            preds.append(pred)
            y = np.append(y, pred)
        return pd.Series(index=future_idx, data=preds, dtype=float)

    a, b = fit_ar1(s.values)
    y_last = float(s.values[-1])
    preds = []
    for _ in range(horizon_q):
        y_next = a + b * y_last
        preds.append(float(y_next))
        y_last = float(y_next)
    return pd.Series(index=future_idx, data=preds, dtype=float)

def build_monthly_fail_probs(pred_stress_q, term_months, mapping_mode, params):
    term_months = int(max(1, term_months))
    q_needed = int(math.ceil(term_months / 3))

    pred = pd.Series(pred_stress_q).astype(float)
    pred = pred.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    if len(pred) < q_needed:
        pad = q_needed - len(pred)
        last_val = float(pred.iloc[-1]) if len(pred) > 0 else 0.0
        last_idx = pred.index[-1] if len(pred.index) > 0 else pd.Period("2020Q1", freq="Q")
        extra_idx = pd.period_range(last_idx + 1, last_idx + pad, freq="Q")
        pred = pd.concat([pred, pd.Series(index=extra_idx, data=[last_val] * pad, dtype=float)])

    pred = pred.iloc[:q_needed]
    s_norm = normalize_01(pred)

    if mapping_mode == "레짐":
        thr_q = float(params["stress_quantile"])
        thr = float(s_norm.quantile(thr_q)) if len(s_norm) > 0 else 0.5
        p_n = float(params["p_fail_normal"])
        p_s = float(params["p_fail_stress"])
        p_q = np.where(s_norm.values >= thr, p_s, p_n).astype(float)
    else:
        p0 = float(params["p_fail_base"])
        sens = float(params["sensitivity"])
        pmax = float(params["p_fail_max"])
        p_q = np.clip(p0 + sens * s_norm.values, 0.0, pmax).astype(float)

    p_q = np.nan_to_num(p_q, nan=float(params.get("p_fail_normal", 0.03)))
    p_month = np.repeat(p_q, 3)[:term_months]
    return p_month, s_norm

def fv_all_success(target, initial, r, term, compound):
    term = int(max(1, term))
    if compound:
        if r == 0:
            denom = float(term)
        else:
            denom = float(((1 + r) ** term - 1) / r)
        numer = float(target - initial * (1 + r) ** term)
        return max(0.0, numer / denom)
    denom = float(term + r * (term - 1) * term / 2.0)
    numer = float(target - initial * (1 + r * term))
    if denom <= 0:
        return float("nan")
    return max(0.0, numer / denom)

@st.cache_data(show_spinner=False)
def simulate(term_months, n_sims, initial, p_deposit, r, compound, p_fail_month, seed=11):
    rng = np.random.default_rng(seed)
    term_months = int(term_months)
    n_sims = int(n_sims)

    p_fail = np.array(p_fail_month, dtype=float)
    p_fail = np.nan_to_num(p_fail, nan=0.0)
    p_fail = np.clip(p_fail, 0.0, 1.0)

    final_bal = np.zeros(n_sims, dtype=float)
    success_deposits = np.zeros(n_sims, dtype=int)

    for i in range(n_sims):
        bal = float(initial)
        principal = float(initial)
        interest_bucket = 0.0

        for t in range(term_months):
            if compound:
                bal = bal * (1 + r)
            else:
                interest_bucket += principal * r
                bal = principal + interest_bucket

            ok = rng.random() >= float(p_fail[t])
            if ok:
                principal += p_deposit
                success_deposits[i] += 1
                if compound:
                    bal += p_deposit
                else:
                    bal = principal + interest_bucket

        final_bal[i] = float(bal)

    return final_bal, success_deposits

def fig_consumption_plotly(consumption):
    df = pd.DataFrame({"q": consumption.index.astype(str), "consumption": consumption.values})
    fig = px.line(df, x="q", y="consumption", title="분기 소비지출(원자료)")
    fig.update_layout(xaxis_title="분기", yaxis_title="소비지출", hovermode="x unified")
    return fig

def fig_stress_forecast_plotly(stress, in_sample_forecast, future_forecast, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stress.index.astype(str), y=stress.values, mode="lines", name="Stress(변동성)"))
    if in_sample_forecast is not None and len(in_sample_forecast.dropna()) > 0:
        s = in_sample_forecast.dropna()
        fig.add_trace(go.Scatter(x=s.index.astype(str), y=s.values, mode="lines", name="Rolling Forecast(표본 내)"))
    if future_forecast is not None and len(future_forecast.dropna()) > 0:
        fig.add_trace(go.Scatter(x=future_forecast.index.astype(str), y=future_forecast.values, mode="lines", name="Forecast(미래 분기)"))
    fig.update_layout(title=title, xaxis_title="분기", yaxis_title="Stress / Forecast", hovermode="x unified")
    return fig

def fig_monthly_fail_plotly(p_fail_month):
    x = np.arange(1, len(p_fail_month) + 1)
    df = pd.DataFrame({"월": x, "p_fail": p_fail_month})
    fig = px.line(df, x="월", y="p_fail", title="월 납입 실패확률(가입기간 내)")
    fig.update_layout(xaxis_title="월", yaxis_title="실패확률", hovermode="x unified")
    return fig

def fig_success_curve_plotly(p_grid, prob_grid):
    df = pd.DataFrame({"월 납입 가능액 P(원)": p_grid, "목표 달성확률": prob_grid})
    fig = px.line(df, x="월 납입 가능액 P(원)", y="목표 달성확률", title="월 납입 가능액 vs 목표 달성확률")
    fig.update_layout(hovermode="x unified")
    return fig

def fig_final_hist_plotly(final_bal, target):
    df = pd.DataFrame({"만기금액": final_bal})
    fig = px.histogram(df, x="만기금액", nbins=40, title="만기금액 분포")
    fig.add_vline(x=target)
    fig.update_layout(xaxis_title="만기금액(원)", yaxis_title="빈도")
    return fig

st.title("적금 목표 달성 확률 시뮬레이터")

with st.sidebar:
    st.subheader("0) 데이터")
    st.write(f"프로젝트 폴더 내 파일 자동 참조: {DATA_FILE}")
    st.caption("파일은 stack.py와 같은 폴더에 있어야 합니다.")

    st.subheader("1) 스트레스 산출")
    vol_window = st.number_input(
        "변동성 rolling window(분기)",
        min_value=4, max_value=40, value=8, step=1,
        help="소비지출의 로그차분(성장률 유사) 변동성을 계산할 때 사용하는 분기 단위 창 길이입니다. 예: 8이면 최근 8분기의 변동성(표준편차)을 봅니다."
    )

    st.subheader("2) 예측 모델 선택(체크)")
    use_rm = st.checkbox("Rolling Mean 모델", value=True, help="최근 k분기 스트레스 평균으로 다음 분기 스트레스를 예측합니다.")
    use_ar1 = st.checkbox("AR(1) 모델", value=False, help="스트레스(t+1) = a + b*스트레스(t) 형태의 단순 자기회귀 모델입니다.")

    rm_k = st.number_input(
        "Rolling Mean 예측 window(분기)",
        min_value=2, max_value=40, value=8, step=1,
        help="Rolling Mean 모델에서 평균을 낼 때 사용하는 최근 분기 개수(k)입니다."
    )
    min_train = st.number_input(
        "최소 학습 구간(분기)",
        min_value=6, max_value=80, value=12, step=1,
        help="Rolling 예측을 시작하기 위해 필요한 최소 과거 데이터 길이입니다. 너무 크면 예측이 비어 NaN이 생길 수 있습니다."
    )

    st.subheader("3) 스트레스 → 실패확률 매핑")
    mapping_mode = st.radio(
        "매핑 방식",
        ["레짐", "연속"],
        index=0,
        help="레짐: 스트레스가 일정 기준 이상이면 실패확률을 높게 적용. 연속: 스트레스가 커질수록 실패확률이 선형으로 증가."
    )

    if mapping_mode == "레짐":
        stress_quantile = st.number_input(
            "스트레스 기준(상위 분위, 0~1)",
            min_value=0.50, max_value=0.99, value=0.80, step=0.01,
            help="정규화된 스트레스가 이 분위수 기준 이상이면 '스트레스 구간'으로 분류합니다. 예: 0.8이면 상위 20%가 스트레스."
        )
        p_fail_normal = st.number_input(
            "정상 구간 월 실패확률(0~1)",
            min_value=0.0, max_value=0.50, value=0.03, step=0.01,
            help="정상 구간에서 한 달 납입이 실패할 확률입니다."
        )
        p_fail_stress = st.number_input(
            "스트레스 구간 월 실패확률(0~1)",
            min_value=0.0, max_value=0.95, value=0.15, step=0.01,
            help="스트레스 구간에서 한 달 납입이 실패할 확률입니다. 정상보다 크게 두는 것이 일반적입니다."
        )
        mapping_params = {
            "stress_quantile": float(stress_quantile),
            "p_fail_normal": float(p_fail_normal),
            "p_fail_stress": float(p_fail_stress),
        }
    else:
        p_fail_base = st.number_input(
            "기본 월 실패확률(0~1)",
            min_value=0.0, max_value=0.50, value=0.03, step=0.01,
            help="스트레스가 0에 가까운 상황에서의 실패확률 기준값입니다."
        )
        sensitivity = st.number_input(
            "민감도(스트레스→실패확률)",
            min_value=0.0, max_value=1.50, value=0.20, step=0.01,
            help="정규화된 스트레스(0~1)가 커질 때 실패확률이 얼마나 증가하는지 결정합니다."
        )
        p_fail_max = st.number_input(
            "실패확률 상한(0~1)",
            min_value=0.05, max_value=0.99, value=0.40, step=0.01,
            help="연속 매핑에서 실패확률이 이 값을 넘지 않도록 제한합니다."
        )
        mapping_params = {
            "p_fail_base": float(p_fail_base),
            "sensitivity": float(sensitivity),
            "p_fail_max": float(p_fail_max),
        }

    st.subheader("4) 적금 입력")
    target = st.number_input(
        "목표 금액(원)",
        min_value=0, value=5_000_000, step=100_000,
        help="만기 시점에 달성하고 싶은 목표 금액입니다."
    )
    initial = st.number_input(
        "초기 보유액(원)",
        min_value=0, value=0, step=100_000,
        help="시작 시점에 이미 보유한 금액입니다."
    )
    term_months = st.number_input(
        "가입기간(개월)",
        min_value=1, max_value=120, value=12, step=1,
        help="적금 납입 기간(개월)입니다. 분기 예측치는 3개월 단위로 월에 적용됩니다."
    )

    monthly_rate_pct = st.number_input(
        "월 이자율(%)",
        min_value=0.0, max_value=10.0, value=0.3, step=0.1,
        help="월 이자율을 %로 입력합니다. 예: 0.3은 0.3% (소수로는 0.003)입니다."
    )
    monthly_rate = float(monthly_rate_pct) / 100.0

    compound = st.toggle(
        "복리 적용",
        value=True,
        help="복리 ON: 매달 잔고에 이자를 적용한 뒤 납입(성공 시)을 더합니다. OFF: 단리 형태로 근사합니다."
    )

    st.subheader("5) 시뮬레이션 설정")
    n_sims = st.selectbox(
        "시뮬레이션 경로 수",
        [2000, 5000, 10000, 20000],
        index=2,
        help="많을수록 안정적이지만 느려집니다. 10,000 권장."
    )
    desired_prob = st.number_input(
        "원하는 달성확률(0~1)",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
        help="이 달성확률을 만족하기 위한 추천 월 납입 가능액 P를 역산합니다."
    )

    run = st.button("계산 실행", help="버튼을 눌렀을 때만 시뮬레이션을 수행합니다.")

try:
    consumption = load_local_quarterly_csv()
    stress = build_stress_from_consumption(consumption, vol_window=vol_window)
except Exception as e:
    st.error(str(e))
    st.stop()

models_selected = []
if use_rm:
    models_selected.append("Rolling Mean")
if use_ar1:
    models_selected.append("AR(1)")
if len(models_selected) == 0:
    models_selected = ["Rolling Mean"]

active_model = models_selected[0]
if len(models_selected) > 1:
    active_model = st.sidebar.selectbox(
        "적용 모델 선택",
        models_selected,
        index=0,
        help="체크한 모델이 2개 이상이면, 실제 적용할 1개 모델을 선택합니다."
    )

forecast_rm = rolling_forecast_rolling_mean(stress, k=rm_k, min_train=min_train) if "Rolling Mean" in models_selected else None
forecast_ar1 = rolling_forecast_ar1(stress, min_train=min_train) if "AR(1)" in models_selected else None
in_sample = forecast_rm if active_model == "Rolling Mean" else forecast_ar1

term_quarters = int(math.ceil(int(term_months) / 3))
future_q = forecast_future_quarters(stress=stress, model_name=active_model, horizon_q=term_quarters, rm_k=rm_k)

p_fail_month, used_future_norm = build_monthly_fail_probs(
    pred_stress_q=future_q,
    term_months=int(term_months),
    mapping_mode=mapping_mode,
    params=mapping_params
)

p_all = fv_all_success(target=float(target), initial=float(initial), r=float(monthly_rate), term=int(term_months), compound=bool(compound))
if not np.isfinite(p_all):
    st.error("입력값으로 월 납입액 계산이 불가능합니다. 기간/금리/목표를 확인해 주세요.")
    st.stop()

p_all_int = int(round(float(p_all)))
p_max_for_input = int(max(100_000, p_all_int * 3))

p_deposit = st.sidebar.number_input(
    "월 납입 가능액 P(원)",
    min_value=0,
    max_value=p_max_for_input,
    value=int(max(0, p_all_int)),
    step=10_000,
    help="한 달에 실제로 납입할 수 있다고 가정하는 금액입니다. 미납이 발생하면 해당 월은 0원이 들어갑니다."
)

st.subheader("현재 설정 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric("선택 모델", active_model)
c2.metric("매핑 방식", mapping_mode)
c3.metric("전기간 납입 성공 시 필요 P", f"{p_all_int:,} 원")
c4.metric("월 납입 가능액 P", f"{int(p_deposit):,} 원")

tab1, tab2, tab3 = st.tabs(["결과 요약", "확률/분포", "스트레스/리스크 근거"])

with tab1:
    st.info("이 페이지는 입력 조건에서의 핵심 결과(성공률/미납 기대치/추천 P)를 요약합니다. 거시 지표를 개인 리스크의 프록시로 사용하는 시뮬레이터입니다.")

    if run:
        final_bal, ok_cnt = simulate(
            term_months=int(term_months),
            n_sims=int(n_sims),
            initial=float(initial),
            p_deposit=float(p_deposit),
            r=float(monthly_rate),
            compound=bool(compound),
            p_fail_month=p_fail_month
        )

        success_prob = float(np.mean(final_bal >= float(target)))
        exp_miss = float(int(term_months) - np.mean(ok_cnt))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("목표 달성확률", f"{success_prob * 100:.1f}%")
        c6.metric("평균 미납 개월", f"{exp_miss:.2f} 개월")
        c7.metric("만기금액 중앙값", f"{int(np.median(final_bal)):,} 원")
        c8.metric("만기금액 하위10%", f"{int(np.quantile(final_bal, 0.10)):,} 원")

        lo, hi = 0.0, float(max(p_all * 3.0, 100000.0))
        for _ in range(12):
            mid = (lo + hi) / 2.0
            fb, _ = simulate(
                term_months=int(term_months),
                n_sims=int(min(7000, int(n_sims))),
                initial=float(initial),
                p_deposit=float(mid),
                r=float(monthly_rate),
                compound=bool(compound),
                p_fail_month=p_fail_month,
                seed=19
            )
            pr = float(np.mean(fb >= float(target)))
            if pr >= float(desired_prob):
                hi = mid
            else:
                lo = mid

        st.success(f"원하는 달성확률 {int(float(desired_prob)*100)}% 기준 추천 월 납입 가능액(근사): {int(round(hi)):,} 원")
    else:
        st.write("사이드바에서 값을 조정한 뒤 **계산 실행**을 눌러 결과를 확인하세요.")

with tab2:
    st.info("만기금액 분포에서 **목표선 오른쪽 면적**이 넓을수록 목표 달성확률이 높습니다. 분포가 넓을수록 결과 불확실성이 큽니다.")

    if run:
        final_bal, ok_cnt = simulate(
            term_months=int(term_months),
            n_sims=int(n_sims),
            initial=float(initial),
            p_deposit=float(p_deposit),
            r=float(monthly_rate),
            compound=bool(compound),
            p_fail_month=p_fail_month
        )

        fig_hist = fig_final_hist_plotly(final_bal, float(target))
        st.plotly_chart(fig_hist, use_container_width=True)

        p_center = float(p_all)
        p_min = float(max(0.0, p_center * 0.5))
        p_max = float(max(p_center * 1.8, p_center + 10000.0))
        grid = np.linspace(p_min, p_max, 18)

        probs = []
        for p in grid:
            fb, _ = simulate(
                term_months=int(term_months),
                n_sims=int(min(4000, int(n_sims))),
                initial=float(initial),
                p_deposit=float(p),
                r=float(monthly_rate),
                compound=bool(compound),
                p_fail_month=p_fail_month,
                seed=23
            )
            probs.append(float(np.mean(fb >= float(target))))

        fig_curve = fig_success_curve_plotly(grid, probs)
        st.plotly_chart(fig_curve, use_container_width=True)

        out = pd.DataFrame({"final_amount": final_bal, "success_deposit_months": ok_cnt})
        st.download_button(
            "시뮬레이션 결과 CSV 다운로드",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="simulation_results.csv",
            mime="text/csv"
        )
    else:
        st.write("사이드바에서 **계산 실행**을 눌러 결과를 생성하세요.")

with tab3:
    st.info(
        "모델은 분기 스트레스(변동성)를 예측하는 방식이며, 매핑은 예측 스트레스를 월 납입 실패확률로 변환하는 규칙입니다. "
        "소비지출(원자료)은 스케일이 커서 별도 그래프로 분리해 표시합니다."
    )

    st.plotly_chart(fig_consumption_plotly(consumption), use_container_width=True)

    title = f"스트레스/예측 | 모델: {active_model} | 매핑: {mapping_mode}"
    fig_sf = fig_stress_forecast_plotly(stress, in_sample, future_q, title)
    st.plotly_chart(fig_sf, use_container_width=True)

    fig_pf = fig_monthly_fail_plotly(p_fail_month)
    st.plotly_chart(fig_pf, use_container_width=True)

    st.write("가입기간 내 월별 실패확률(p_fail)")
    st.dataframe(
        pd.DataFrame({"월": np.arange(1, int(term_months) + 1), "p_fail": p_fail_month}),
        use_container_width=True
    )
