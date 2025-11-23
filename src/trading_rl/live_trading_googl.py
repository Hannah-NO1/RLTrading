import os
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from trading_rl.position_sizing import WinRatioModulatedSizer

# =========================
# 기본 설정
# =========================

SYMBOL = "GOOGL"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "googl_ppo"

# 프로젝트 루트: .../rl-trading-googl
BASE_DIR = Path(__file__).resolve().parents[2]

# 로그 디렉토리: .../rl-trading-googl/logs
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 라이브 트레이딩에서 사용할 seed
set_random_seed(42)


# =========================
# CSV 로그 헬퍼
# =========================


def append_live_log_row(
    timestamp,
    symbol,
    price,
    action,
    position_before,
    position_after,
    trade_size,
    equity_before,
    equity_after,
    max_position_size,
    leverage,
    rolling_win_ratio=None,
    dynamic_risk_fraction=None,
    mode="paper",
):
    """
    라이브 트레이딩 결과를 매 줄 CSV로 저장하는 함수.
    날짜별로 live_trading_YYYY-MM-DD.csv 파일 생성/추가.
    """
    date_str = timestamp.date().isoformat()
    log_file = LOG_DIR / f"live_trading_{date_str}.csv"

    file_exists = log_file.exists()

    with log_file.open("a", newline="") as f:
        writer = csv.writer(f)

        # 파일이 처음 생성되면 헤더를 한 번만 씀
        if not file_exists:
            writer.writerow(
                [
                    "timestamp_utc",
                    "symbol",
                    "mode",
                    "price",
                    "action",
                    "position_before",
                    "position_after",
                    "trade_size",
                    "equity_before",
                    "equity_after",
                    "max_position_size",
                    "leverage",
                    "rolling_win_ratio",
                    "dynamic_risk_fraction",
                ]
            )

        writer.writerow(
            [
                timestamp.isoformat(),
                symbol,
                mode,
                float(price),
                int(action),
                int(position_before),
                int(position_after),
                int(trade_size),
                float(equity_before),
                float(equity_after),
                int(max_position_size),
                float(leverage),
                (float(rolling_win_ratio) if rolling_win_ratio is not None else ""),
                (
                    float(dynamic_risk_fraction)
                    if dynamic_risk_fraction is not None
                    else ""
                ),
            ]
        )


# =========================
# 시세 / 관측치 유틸
# =========================


def fetch_recent_intraday_data(symbol: str, lookback_minutes: int = 60) -> pd.DataFrame:
    """
    yfinance를 사용해서 최근 intraday(1분봉) 데이터를 가져옴.
    (시장 시간 외에는 값이 잘 안 바뀔 수 있음)
    """
    # 1일치 1분봉 데이터
    df = yf.download(
        symbol,
        period="1d",
        interval="1m",
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError(f"No intraday data for {symbol}")

    df = df.tail(lookback_minutes)
    return df


def build_observation(
    price_window: np.ndarray,
    equity: float,
    position: int,
    max_position_size: int,
    obs_dim: int,
) -> np.ndarray:
    """
    학습 시 사용했던 관측 차원(obs_dim)에 맞게
    - 최근 종가 윈도우
    - 현재 equity (정규화)
    - 현재 포지션 (정규화)
    등을 벡터로 구성해서 반환.

    여기서는 간단히:
    - 최근 20개 종가 변화율
    - equity / 100000
    - position / max(1, max_position_size)
    로 구성한 후 obs_dim에 맞게 패딩/자르기 한다.
    """
    closes = price_window.astype(float)

    if len(closes) < 2:
        returns = np.zeros(20)
    else:
        rets = np.diff(closes) / closes[:-1]
        if len(rets) >= 20:
            returns = rets[-20:]
        else:
            returns = np.pad(rets, (20 - len(rets), 0))

    # 특성 구성
    feat_equity = equity / 100000.0
    denom_pos = max(1, max_position_size)
    feat_position = position / denom_pos

    features = np.concatenate(
        [
            returns,  # 20
            np.array([feat_equity, feat_position], dtype=float),  # 2
        ]
    )  # 길이 22

    # obs_dim에 맞추기 (패딩 또는 자르기)
    if len(features) < obs_dim:
        features = np.pad(features, (0, obs_dim - len(features)))
    else:
        features = features[:obs_dim]

    return features.astype(np.float32)


# =========================
# 라이브 트레이딩 메인
# =========================


def run_live_trading():
    """
    학습된 PPO 모델을 사용해서
    - 실시간(또는 1분 간격)으로 GOOGL 시세를 가져오고
    - RL 정책이 액션을 내리면
      포지션/자본을 업데이트하고
    - 그 내역을 CSV로 기록하는 루프.
    실제 주문은 전혀 보내지 않음(완전 페이퍼 모드).
    """

    # 1. 모델 로드
    print(f"📂 학습된 모델 로딩 중... ({MODEL_PATH})")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델이 존재하지 않습니다: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH)
    print("✅ 모델 로딩 완료")

    # 관측 차원 정보 (정책이 기대하는 input 크기)
    try:
        obs_dim = int(model.observation_space.shape[0])
    except Exception:
        # 일부 버전에서는 model.policy.observation_space 사용 필요
        obs_dim = int(model.policy.observation_space.shape[0])
    print(f"🧠 모델 관측 차원: {obs_dim}")

    # 2. 초기 상태 설정
    # - 여기서는 StockTrak 가상 계좌 상태와 맞추기 위해
    #   equity, 보유 주식 수를 수동 입력하거나,
    #   나중에 CSV에서 읽어와도 됨.
    #
    # 예시: 네가 보여준 포트폴리오 상태 기준
    #   - Portfolio Value: 10304.03
    #   - GOOGL 34주 보유
    #   - Cash ≈ 457.97 (대략)
    initial_equity = 10304.03
    position = 34

    # 최근 종가 하나를 가져와서 대략적인 현금 추정
    intraday = fetch_recent_intraday_data(SYMBOL, lookback_minutes=60)
    last_price = float(intraday["Close"].iloc[-1])
    cash = initial_equity - position * last_price

    equity = initial_equity

    print(
        f"▶ 시작 상태: equity={equity:.2f}, position={position}, "
        f"price={last_price:.2f}, cash≈{cash:.2f}"
    )
    print("🚀 라이브 트레이딩 루프 시작 (페이퍼 모드)")
    print("    Ctrl+C 로 종료할 수 있습니다.\n")

    # 3. 포지션 사이저 (승률 연동 + 동적 레버리지)
    trade_history = []  # 각 step마다 (equity 증가: 1, 감소: -1)
    sizer = WinRatioModulatedSizer(
        lookback_period=60,
        base_fraction=0.015,  # 기본 리스크 비율 (조금 보수적)
        sigmoid_L=2.5,  # 최대 2.5배까지 리스크 확대
        sigmoid_k=15,  # 승률 변화에 대한 민감도
        sigmoid_wr0=0.33,  # 기준 승률
        use_dynamic_leverage=True,
        max_leverage=2.0,
        stop_loss_pips=10,  # 대략 10달러 손절 가정
        pip_value=1.0,  # 1달러당 1단위
    )

    # 4. 라이브 루프
    #    - 여기서는 60초마다 1번씩 정책 실행 (원하면 나중에 5분, 10분으로 바꿀 수 있음)
    try:
        while True:
            try:
                # 현재 시각: UTC 기준
                now_utc = datetime.now(timezone.utc)

                # 1) 최근 시세 가져오기
                intraday = fetch_recent_intraday_data(SYMBOL, lookback_minutes=60)
                closes = intraday["Close"].values.astype(float)
                last_price = float(closes[-1])

                # 2) 사이저로 현재 최대 포지션 상한 계산
                sizing_info = sizer.calculate_size(equity, trade_history)
                max_position_size = max(0, sizing_info.get("position_size", 0))
                leverage = sizing_info.get("leverage", 1.0)

                # 3) 관측 벡터 생성
                obs_vec = build_observation(
                    price_window=closes,
                    equity=equity,
                    position=position,
                    max_position_size=max_position_size if max_position_size > 0 else 1,
                    obs_dim=obs_dim,
                )
                obs = np.expand_dims(obs_vec, axis=0)  # (1, obs_dim)

                # 4) RL 정책으로부터 액션 생성
                action_raw, _ = model.predict(obs, deterministic=True)
                # Discrete 환경에서는 0차원 ndarray가 나올 수 있으므로 int(...)로 변환
                action = int(action_raw)

                # 액션 의미(예시):
                # 0: 관망
                # 1: 포지션 늘리기 (buy)
                # 2: 포지션 줄이기 (sell / reduce)
                position_before = position
                equity_before = equity

                # 5) 목표 포지션 계산 (여기서는 간단히 ±10%씩 조정 예시)
                delta_units = 0
                if max_position_size <= 0:
                    target_position = position  # 사이징이 0이면 포지션 유지
                else:
                    step_units = max(1, max_position_size // 10)  # 전체 상한의 10% 단위
                    if action == 1:  # buy
                        delta_units = step_units
                    elif action == 2:  # sell
                        delta_units = -step_units
                    else:
                        delta_units = 0

                    target_position = position + delta_units
                    # 0 ~ max_position_size 사이로 클램프
                    target_position = max(0, min(max_position_size, target_position))

                trade_size = target_position - position

                # 6) 포지션/캐시/에쿼티 업데이트 (페이퍼 모드)
                if trade_size != 0:
                    trade_cash_flow = -trade_size * last_price  # 사면 cash 감소, 팔면 증가
                    cash += trade_cash_flow
                    position = target_position

                # 자본 = 현금 + 보유주식 가치
                equity = cash + position * last_price

                # 7) 이번 step의 win/loss 기록 (equity 증감 기준)
                equity_delta = equity - equity_before
                if equity_delta > 0:
                    trade_history.append(1)
                elif equity_delta < 0:
                    trade_history.append(-1)
                # 0이면 기록 X (또는 0으로 기록하고 싶으면 append(0))

                # 8) 터미널 로그 출력
                print(
                    f"[LIVE] {now_utc:%Y-%m-%d %H:%M:%S} | "
                    f"price={last_price:.2f}, action={action}, "
                    f"pos {position_before}->{position}, "
                    f"eq {equity_before:.2f}->{equity:.2f} (Δ {equity_delta:.2f}), "
                    f"max_pos={max_position_size}, lev={leverage:.2f}"
                )

                # 9) CSV 로그 기록
                append_live_log_row(
                    timestamp=now_utc,
                    symbol=SYMBOL,
                    price=last_price,
                    action=action,
                    position_before=position_before,
                    position_after=position,
                    trade_size=trade_size,
                    equity_before=equity_before,
                    equity_after=equity,
                    max_position_size=max_position_size,
                    leverage=leverage,
                    rolling_win_ratio=sizing_info.get("rolling_win_ratio"),
                    dynamic_risk_fraction=sizing_info.get("dynamic_risk_fraction"),
                    mode="paper",  # 실제 주문이 아니라 시뮬레이션임
                )

                # 10) 다음 step까지 대기 (60초)
                time.sleep(60)

            except KeyboardInterrupt:
                print("\n🛑 사용자에 의해 라이브 루프가 중단되었습니다.")
                break
            except Exception as e:
                # 에러가 나도 프로그램이 완전히 죽지 않게 하고, 잠시 후 재시도
                print(f"⚠️ 에러 발생: {e}")
                print("10초 후 재시도합니다...")
                time.sleep(10)

    finally:
        print("✅ 라이브 트레이딩 종료.")


if __name__ == "__main__":
    run_live_trading()
