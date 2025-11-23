import numpy as np
import pandas as pd
import yfinance as yf
import os
from .position_sizing import WinRatioModulatedSizer
from .env_trading import RLTradingEnv


# -----------------------------
# 1. GOOGL 데이터 로딩 함수
# -----------------------------
def load_googl(period: str = "5y") -> pd.DataFrame:
    """
    yfinance를 사용해서 GOOGL의 과거 주가 데이터를 가져옵니다.

    :param period: '6mo', '1y', '5y', 'max' 등 yfinance 지원 기간 문자열
    :return: 'close' 컬럼을 가진 DataFrame
    """
    df = yf.download("GOOGL", period=period, auto_adjust=True)
    if df.empty:
        raise ValueError("GOOGL 데이터를 가져오지 못했습니다. 인터넷 연결 또는 yfinance 설정을 확인하세요.")
    # env_trading에서 'close' 컬럼을 사용하므로 이름을 맞춰줍니다.
    df = df[["Close"]].rename(columns={"Close": "close"})
    return df


# -----------------------------
# 2. 성능 지표 계산 함수
# -----------------------------
def compute_performance_metrics(equity_curve: np.ndarray, trading_days_per_year: int = 252):
    """
    자본 곡선을 바탕으로 CAGR, MDD, Sharpe, Calmar 를 계산.
    equity_curve: 각 시점의 equity (길이 N)
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) < 2:
        raise ValueError("equity_curve 길이가 너무 짧습니다.")

    initial = equity_curve[0]
    final = equity_curve[-1]
    n_days = len(equity_curve) - 1

    # 연 수 기준 (일봉 기준으로 252거래일 가정)
    years = n_days / trading_days_per_year if trading_days_per_year > 0 else 1.0

    # CAGR
    if initial > 0 and years > 0:
        cagr = (final / initial) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan

    # 최대 낙폭 (MDD)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    mdd = drawdowns.min()  # 음수 값

    # 일간 수익률
    daily_returns = equity_curve[1:] / equity_curve[:-1] - 1.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(trading_days_per_year)
    else:
        sharpe = np.nan

    # Calmar 비율 (CAGR / |MDD|)
    if mdd < 0:
        calmar = cagr / abs(mdd)
    else:
        calmar = np.nan

    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Calmar": calmar,
        "FinalEquity": final,
        "InitialEquity": initial,
        "NumDays": n_days,
    }


# -----------------------------
# 3. 단일 환경에서 모델 평가 함수
# -----------------------------
def evaluate_model_on_env(env: RLTradingEnv, model) -> np.ndarray:
    """
    학습된 모델을 주어진 환경(env)에서 한 번 실행하여 equity curve 를 리턴.
    """
    obs = env.reset()
    done = False
    equity_curve = [env.equity]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        equity_curve.append(env.equity)

    return np.asarray(equity_curve, dtype=float)


# -----------------------------
# 4. 메인 학습 루프 (Train / Test 분리)
# -----------------------------
def train_ppo():
    # ================
    # (1) 데이터 로딩
    # ================
    df_all = load_googl("5y")

    # 5년 데이터 중에서:
    # - 앞쪽 대부분: Train
    # - 마지막 1년(252거래일): Test 라고 가정
    if len(df_all) <= 252 + 60:
        raise ValueError("데이터가 너무 적습니다. period를 더 길게 하거나 다른 소스를 사용하세요.")

    df_train = df_all.iloc[:-252]   # 앞부분 (학습용)
    df_test = df_all.iloc[-252:]    # 뒷부분 1년 (테스트용)

    # ================
    # (2) 포지션 사이저 설정 (승률 연동 + 동적 레버리지)
    # ================
    sizer_train = WinRatioModulatedSizer(
        lookback_period=60,
        base_fraction=0.015,  # 보수적 기본 리스크
        sigmoid_L=2.5,
        sigmoid_k=15.0,
        sigmoid_wr0=0.33,
        use_dynamic_leverage=True,
        max_leverage=2.0,
        stop_loss_ticks=50,
        tick_value=1.0,
    )

    # 테스트 환경도 같은 사이저 구조 사용 (trade_history 등은 env 내에서 따로 관리)
    sizer_test = WinRatioModulatedSizer(
        lookback_period=60,
        base_fraction=0.015,
        sigmoid_L=2.5,
        sigmoid_k=15.0,
        sigmoid_wr0=0.33,
        use_dynamic_leverage=True,
        max_leverage=2.0,
        stop_loss_ticks=50,
        tick_value=1.0,
    )

    # ================
    # (3) Train / Test 환경 생성
    # ================
    env_train = RLTradingEnv(
        price_df=df_train,
        sizer=sizer_train,
        window_size=30,
        initial_equity=100_000.0,
        tick_value=1.0,
    )

    env_test = RLTradingEnv(
        price_df=df_test,
        sizer=sizer_test,
        window_size=30,
        initial_equity=100_000.0,
        tick_value=1.0,
    )

    # ================
    # (4) PPO로 학습 (Train 환경)
    # ================
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise ImportError(
            "stable-baselines3 가 설치되어 있지 않습니다. "
            "먼저 `pip install stable-baselines3` 를 실행하세요."
        ) from e

    vec_env_train = DummyVecEnv([lambda: env_train])

    model = PPO(
        "MlpPolicy",
        vec_env_train,
        verbose=1,
        tensorboard_log=None,
    )

    # timesteps 는 필요에 따라 조절
    model.learn(total_timesteps=50_000)

    os.makedirs("models", exist_ok=True)
    model.save("models/googl_ppo")


    # ================
    # (5) Test 환경에서 평가
    # ================
    equity_curve_test = evaluate_model_on_env(env_test, model)
    metrics = compute_performance_metrics(equity_curve_test)

    print("\n===== Test 구간 성과 지표 (최근 1년) =====")
    print(f"초기 자본: {metrics['InitialEquity']:.2f}")
    print(f"최종 자본: {metrics['FinalEquity']:.2f}")
    print(f"거래 일수(대략): {metrics['NumDays']}")
    print(f"CAGR (연복리): {metrics['CAGR'] * 100:.2f}%")
    print(f"MDD (최대낙폭): {metrics['MDD'] * 100:.2f}%")
    print(f"Sharpe 비율: {metrics['Sharpe']:.2f}")
    print(f"Calmar 비율: {metrics['Calmar']:.2f}")

    # ================
    # (6) 자본 곡선 시각화
    # ================
    try:
        import matplotlib.pyplot as plt

        plt.plot(equity_curve_test)
        plt.title("Test Equity Curve (GOOGL, 최근 1년)")
        plt.xlabel("Step")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib 미설치: 자본 곡선 그래프는 생략합니다.")


if __name__ == "__main__":
    train_ppo()
