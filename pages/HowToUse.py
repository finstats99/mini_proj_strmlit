import streamlit as st

st.set_page_config(page_title="사용 가이드", layout="wide")

st.title("사용 가이드: 적금 목표 달성 확률 시뮬레이터")

st.write(
    "이 앱은 **분기 소비지출의 흔들림(스트레스)** 를 예측한 뒤, 이를 **월 납입 실패확률**로 변환해 "
    "**목표 달성확률**과 **추천 월 납입 가능액**을 시뮬레이션으로 보여줍니다."
)

with st.expander("✅ 한 줄 요약", expanded=True):
    st.markdown(
        """
- **스트레스(stress)**: 소비지출이 *얼마나 불안정한지*를 나타내는 지표  
- **예측 모델(rolling prediction)**: 미래 분기 스트레스를 단순한 방식으로 예측  
- **매핑(mapping)**: 스트레스(점수)를 **월 실패확률(확률)** 로 바꾸는 규칙  
- **시뮬레이션**: 실패확률에 따라 납입 성공/실패를 반복해 목표 달성확률 계산
"""
    )

st.divider()

st.header("전체 흐름(파이프라인)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("1", "소비지출")
c2.metric("2", "스트레스 산출")
c3.metric("3", "스트레스 예측")
c4.metric("4", "실패확률 매핑")
c5.metric("5", "달성확률 시뮬")

st.caption("핵심은 4번: **스트레스 → 실패확률**로 바꾸는 단계입니다.")

st.divider()

st.header("1) 스트레스(stress)란 무엇인가")

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("개념")
    st.markdown(
        """
스트레스는 **소비지출이 얼마나 흔들리는지(불안정성)** 를 나타냅니다.  
소비가 안정적이면 낮고, 갑자기 출렁이면 높습니다.
"""
    )

    st.subheader("계산 절차")
    st.markdown("1) 소비지출 시계열을 \(C_t\) 라고 둡니다.")
    st.markdown("2) 로그차분(성장률 유사) \(g_t\) 를 계산합니다.")
    st.latex(r"g_t = \log(C_t) - \log(C_{t-1})")

    st.markdown("3) 분기 window \(w\) 에 대한 rolling 표준편차를 스트레스로 정의합니다.")
    st.latex(r"stress_t = \mathrm{std}\left(g_{t-w+1}, \dots, g_t\right)")

with colB:
    st.subheader("해석 팁")
    st.markdown(
        """
- **w(윈도우)가 클수록**: 완만(노이즈 감소), 반응 느림  
- **w(윈도우)가 작을수록**: 민감(즉각 반응), 값이 요동칠 수 있음  

추천 시작값(분기 데이터 기준)
- `w=8` (2년치) 정도를 기본으로 두고
- 더 민감하게 보고 싶으면 `w=4~6`, 더 안정적으로 보고 싶으면 `w=10~12`
"""
    )

    st.info(
        "이 앱에서는 스트레스가 ‘확률’이 아니라 ‘점수’입니다. "
        "그래서 다음 단계에서 **실패확률로 변환(매핑)** 이 필요합니다."
    )

st.divider()

st.header("2) 예측 모델(rolling prediction)")

with st.expander("모델별 정의 보기", expanded=True):
    st.markdown("### Rolling Mean")
    st.write("최근 k개 분기 스트레스 평균으로 다음 분기를 예측합니다.")
    st.latex(r"\hat{s}_{t+1} = \frac{1}{k}\sum_{i=t-k+1}^{t} s_i")

    st.markdown("### AR(1)")
    st.write("직전 스트레스에 선형 의존한다고 가정합니다.")
    st.latex(r"s_{t+1} = a + b s_t")

    st.markdown("### ETS(단순 지수평활, SES)")
    st.write("최근 관측치에 더 가중치를 주는 평활 방식입니다. (α가 클수록 최근값 반영 ↑)")
    st.latex(r"\ell_t = \alpha s_t + (1-\alpha)\ell_{t-1}")
    st.latex(r"\hat{s}_{t+1} = \ell_t")

st.divider()

st.header("3) 스트레스 → 실패확률 매핑(가장 중요)")

st.markdown(
    """
예측된 스트레스는 점수이므로, 시뮬레이션에 쓰기 위해 **월 납입 실패확률 \(p_{fail}\)** 로 바꿉니다.  
앱은 먼저 예측 스트레스를 **0~1로 정규화**해서 해석을 쉽게 합니다.
"""
)

st.latex(r"\tilde{s}_q = \frac{\hat{s}_q - \min(\hat{s})}{\max(\hat{s}) - \min(\hat{s})}")

st.caption("정규화 후 \(\tilde{s}_q\) 는 0(낮은 스트레스) ~ 1(높은 스트레스) 범위입니다.")

tab_regime, tab_cont = st.tabs(["레짐(Regime) 매핑", "연속(Continuous) 매핑"])

with tab_regime:
    st.subheader("아이디어")
    st.markdown(
        """
스트레스를 **정상/스트레스 두 상태로 나눠** 실패확률을 “두 단계”로 부여합니다.  
(경계에서 확률이 **뚝** 바뀌는 대신, 설명이 가장 직관적입니다.)
"""
    )

    st.subheader("수식")
    st.write("임계값을 분위수로 정합니다.")
    st.latex(r"\tau = Q_{q}(\tilde{s})")

    st.write("분기별 실패확률은 다음과 같습니다.")
    st.latex(
        r"""
p_{fail,q} =
\begin{cases}
p_{stress}, & \tilde{s}_q \ge \tau \\
p_{normal}, & \tilde{s}_q < \tau
\end{cases}
"""
    )

    st.subheader("월 단위 적용")
    st.markdown(
        """
데이터는 분기 단위이므로 **1분기 = 3개월**에 같은 확률을 반복 적용합니다.  
예: 어떤 분기의 \(p_{fail,q}\)=0.12 라면, 그 분기의 3개월은 모두 0.12로 적용됩니다.
"""
    )

with tab_cont:
    st.subheader("아이디어")
    st.markdown(
        """
스트레스가 커질수록 실패확률이 **점진적으로 증가**하도록 만듭니다.  
(경계에서 뚝 끊기지 않아 더 ‘부드러운’ 리스크 해석이 가능합니다.)
"""
    )

    st.subheader("수식")
    st.latex(r"p_{fail,q} = \mathrm{clip}(p_0 + \beta \tilde{s}_q,\ 0,\ p_{max})")

    st.markdown(
        """
- \(p_0\): 기본 실패확률(스트레스가 낮을 때)  
- \(\beta\): 민감도(스트레스가 0→1로 갈 때 확률이 얼마나 증가하는지)  
- \(p_{max}\): 실패확률 상한  
- \(\mathrm{clip}(\cdot)\): 0과 \(p_{max}\) 사이로 잘라주는 함수
"""
    )

    st.subheader("월 단위 적용")
    st.markdown("레짐과 동일하게, 분기 확률을 3개월에 반복 적용합니다.")

st.divider()

st.header("4) 시뮬레이션(몬테카를로)")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(
        """
매월 납입 성공/실패를 확률로 결정합니다.  
실패확률이 \(p_{fail,m}\) 이면 성공확률은 \(1-p_{fail,m}\) 입니다.
"""
    )
    st.latex(r"I_m \sim \mathrm{Bernoulli}(1 - p_{fail,m})")

with col2:
    st.markdown(
        """
- \(I_m=1\): 납입 성공 → \(P\)가 들어감  
- \(I_m=0\): 납입 실패 → 0원이 들어감  

이 과정을 여러 번 반복해 “만기금액 분포”와 “목표 달성확률”을 계산합니다.
"""
    )

st.divider()

st.header("추천 사용 순서(빠른 체험)")
st.markdown(
    """
1) 목표금액/기간/월이자율/월 납입 가능액 입력  
2) 매핑은 **레짐**으로 시작(직관적)  
3) 정상/스트레스 실패확률을 조절하며 성공확률 변화 확인  
4) 연속으로 바꿔 **민감도(β)** 조절이 결과에 주는 영향 확인  
5) Rolling Mean / AR(1) / ETS를 바꿔 예측 차이가 성공률에 주는 영향 비교
"""
)

st.caption("Tip: 이 앱의 핵심은 ‘정답’이 아니라, 가정(매핑/모델/기간/금리)에 대한 결과 민감도를 빠르게 탐색하는 것입니다.")
