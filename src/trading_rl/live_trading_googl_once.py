import os
import math
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from stable_baselines3 import PPO

# 상대 임포트 (src.trading_rl 패키지 안에서 실행됨)
from src.trading_rl.position_sizing import WinRatioModulatedSizer

# ================================
# 경로 / 고정 설정
# ================================

# 프로젝트 루트 (.../rl-trading-googl)
BASE_DIR = Path(__file__).resolve().parents[2]

# PPO 모델 경로 (학습 시 사용했던 것과 동일해야 함)
MODEL_PATH = BASE_DIR / "models" / "googl_ppo"

# 스케일러가 있다면 여기 (선택 사항, 없으면 없어도 됨)
SCALER_PATH = BASE_DIR / "models" / "googl_scaler.pkl"

# 로그 디렉토리
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ================================
# 사용자가 매일 업데이트할 값들
# (StockTrak 포트폴리오 기준)
# ================================

SYMBOL = "GOOGL"

# 예시: 2025-11-20 기준 포트폴리오 상황
#   - Portfolio Value: 10,304.03
#   - GOOGL 34주 보유, 나머지는 현금
CURRENT_EQUITY_USD = 10304.03   # 현재 포트폴리오 총 가치 (StockTrak 화면 기준)
CURRENT_POSITION_SHARES = 34    # 현재 보유 GOOGL 주식 수

# ================================
# CSV 로그 헬퍼
# ================================

def append_live_log_row(
    timestamp: datetime,
    symbol: str,
    price: float,
    action: int,
    position_before: int,
    position_after: int,
    trade_size: int,
    equity_before: float,
    equity_after: float,
    max_position_size: int,
    leverage: float,
    rolling_win_ratio: Optional[float] = None,
    dynamic_risk_fraction: Optional[float] = None,
    mode: str = "daily_once",
) -> None:
    """
    하루에 한 번 실행되는 RL 의사결정을 CSV로 저장.
    날짜별로 live_trading_YYYY-MM-DD.csv 파일 생성/추가.
    """
    date_str = timestamp.date().isoformat()
    log_file = LOG_DIR / f"live_trading_{date_str}.csv"

    file_exists = log_file.exists()

    with log_file.open("a", newline="") as f:
        writer = csv.writer(f)

        # 파일이 처음 생성되면 헤더 한 번만 쓰기
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
                (float(dynamic_risk_fraction) if dynamic_risk_fraction is not None else ""),
            ]
        )


# ================================
# 유틸: 스케일러 로드 (있을 때만)
# ================================

def load_scaler() -> Optional[object]:
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"✅ 스케일러 로딩 완료: {SCALER_PATH}")
            return scaler
        except Exception as e:
            print(f"⚠️ 스케일러 로드 실패 (무시하고 진행): {e}")
    else:
        print("ℹ️ 스케일러 파일 없음 (SCALER_PATH), 스케일링 없이 진행")
    return None


# ================================
# 유틸: yfinance에서 최근 데이터로 관측 벡터 만들기
# ================================

def build_observation_from_yfinance(
    obs_dim: int,
    symbol: str,
    current_position: int,
    current_equity: float,
) -> Tuple[np.ndarray, float]:
    """
    yfinance에서 최근 일봉 데이터를 받아 PPO가 기대하는 크기의 관측 벡터(obs)를 만든다.
    - obs_dim: PPO 모델 관측 차원 (예: 32)
    - 마지막 2칸은 [포지션 비율, 캐시 비율]로 사용
    """

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)  # 최근 6개월치에서 사용

    df = yf.download(
        symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        progress=False,
    )

    if df.empty:
        raise RuntimeError("yfinance에서 데이터를 가져오지 못했습니다.")

    closes = df["Close"].dropna()

    # 최근 종가
    last_price = float(closes.iloc[-1])

    # 단순 수익률(feature) 계산 -> numpy 1D 배열로 변환 (초강력 방어)
    raw_returns = closes.pct_change().fillna(0.0).values
    returns = np.asarray(raw_returns, dtype=np.float32).reshape(-1)

    # DEBUG (필요하면 확인용)
    # print(f"[DEBUG] obs_dim={obs_dim}, n_returns={len(returns)}")

    # obs 벡터 초기화
    obs = np.zeros(int(obs_dim), dtype=np.float32)

    # 가격 정보에 쓸 수 있는 슬롯 수 (마지막 2칸은 포지션/캐시 비율용)
    if obs_dim > 2:
        price_slots = obs_dim - 2
    else:
        price_slots = obs_dim  # 혹시라도 obs_dim이 2 이하인 경우 방어

    # 실제로 넣을 리턴 개수 = 리턴 길이와 price_slots 중 작은 값
    k = min(len(returns), price_slots)

    if k > 0:
        try:
            obs[:k] = returns[-k:]
        except ValueError as e:
            # 혹시라도 브로드캐스트 에러가 나면 k를 줄여서 한 번 더 시도
            print(
                f"⚠️ obs broadcast 에러: obs.shape={obs.shape}, "
                f"returns.shape={returns.shape}, k={k}, err={e}"
            )
            k2 = min(k, obs.shape[0], returns.shape[0])
            if k2 > 0:
                obs[:k2] = returns[-k2:]

    # 포지션/캐시 비율 계산 (대략)
    position_value = current_position * last_price
    cash_estimate = max(current_equity - position_value, 0.0)

    if current_equity <= 0:
        pos_ratio = 0.0
        cash_ratio = 0.0
    else:
        pos_ratio = position_value / current_equity
        cash_ratio = cash_estimate / current_equity

    # obs 마지막 2칸에 비율 정보 저장 (obs_dim이 2 이하인 경우도 방어)
    if obs_dim >= 1:
        obs[-2 if obs_dim >= 2 else -1] = pos_ratio
    if obs_dim >= 2:
        obs[-1] = cash_ratio

    return obs, last_price


# ================================
# 메인: 하루에 한 번 실행하는 라이브 의사결정
# ================================

def run_live_trading_once() -> None:
    print(f"📂 학습된 모델 로딩 중... ({MODEL_PATH})")

    if not MODEL_PATH.exists() and not (MODEL_PATH.with_suffix(".zip")).exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH} 또는 {MODEL_PATH}.zip")

    # PPO 모델 로드
    model = PPO.load(str(MODEL_PATH), device="cpu")
    print("✅ 모델 로딩 완료")

    # 관측 차원 확인
    try:
        obs_shape = model.observation_space.shape
        if len(obs_shape) == 0:
            obs_dim = 32
        else:
            obs_dim = int(obs_shape[0])
    except Exception:
        obs_dim = 32  # 혹시 실패하면 기본값
    print(f"🧠 모델 관측 차원: {obs_dim}")

    # 스케일러 로드 (선택)
    scaler = load_scaler()

    # ----- 현재 계좌 상태 세팅 (사용자가 위에서 값 수정) -----
    equity = float(CURRENT_EQUITY_USD)
    position = int(CURRENT_POSITION_SHARES)

    # 최근 데이터로 관측 벡터 만들기
    obs_vec, last_price = build_observation_from_yfinance(
        obs_dim=obs_dim,
        symbol=SYMBOL,
        current_position=position,
        current_equity=equity,
    )

    # 필요하다면 스케일링
    if scaler is not None:
        try:
            obs_input = scaler.transform(obs_vec.reshape(1, -1))
        except Exception as e:
            print(f"⚠️ 스케일링 실패, 원본 관측 사용: {e}")
            obs_input = obs_vec.reshape(1, -1)
    else:
        obs_input = obs_vec.reshape(1, -1)

    # ----- RL 에이전트에게 액션 물어보기 -----
    action, _states = model.predict(obs_input, deterministic=True)
    try:
        action = int(action)
    except Exception:
        action = int(np.array(action).flatten()[0])

    # ----- 포지션 사이징 로직 (WinRatioModulatedSizer 사용) -----

    # 여기서는 trade_history를 관리하지 않으니,
    # "데이터 부족" 상태로 인식하게 두고 기본 리스크 비율만 사용
    trade_history: List[int] = []

    sizer = WinRatioModulatedSizer(
        lookback_period=60,
        base_fraction=0.015,   # 대략 자본의 1.5%를 기본 리스크
        sigmoid_L=2.0,
        sigmoid_k=10,
        sigmoid_wr0=0.33,
        use_dynamic_leverage=True,
        max_leverage=2.0,
        stop_loss_pips=50,     # 여기서는 "주당 위험 50달러" 정도의 상징적인 값
        pip_value=1.0,
    )

    sizing_info: Dict = sizer.calculate_size(equity, trade_history)
    max_position_size = int(sizing_info.get("position_size", 0))
    leverage = float(sizing_info.get("leverage", 1.0))
    rolling_win_ratio = sizing_info.get("rolling_win_ratio", None)
    dynamic_risk_fraction = sizing_info.get("dynamic_risk_fraction", None)

    # 최대 매수 가능한 주식 수 (레버리지 고려한 이론치)
    max_affordable_shares = int((equity * leverage) / last_price) if last_price > 0 else 0

    # 실제 타겟 포지션: RL action + 사이저 + 계좌 제약 결합
    if action == 0:
        # HOLD
        target_position = position
    elif action == 1:
        # BUY: 사이저가 제안한 포지션과 계좌 한도를 모두 반영
        target_position = min(max_position_size, max_affordable_shares)
        # 이미 더 많이 들고 있으면 줄이지 않고 유지
        target_position = max(target_position, position)
    elif action == 2:
        # SELL: 전량 청산
        target_position = 0
    else:
        target_position = position

    trade_size = target_position - position

    # 하루에 한 번 "결정"만 내리는 것이므로,
    # 이 스크립트 안에서는 equity를 바꾸지 않고 그대로 사용
    equity_before = equity
    equity_after = equity  # 실제로는 다음 날 가격 변동과 함께 외부(StockTrak)가 업데이트해줌

    now_utc = datetime.now(timezone.utc)

    # ----- 터미널에 요약 출력 -----
    action_str = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(action, f"UNKNOWN({action})")
    print()
    print("===== RL 일일 의사결정 결과 =====")
    print(f"시간 (UTC)       : {now_utc:%Y-%m-%d %H:%M:%S}")
    print(f"심볼            : {SYMBOL}")
    print(f"현재 가격       : {last_price:.2f} USD")
    print(f"현재 포지션     : {position} 주")
    print(f"현재 Equity     : {equity_before:.2f} USD")
    print(f"RL 액션         : {action_str} (raw={action})")
    print(f"사이저 max_pos  : {max_position_size} 주 (leverage={leverage:.2f})")
    print(f"목표 포지션     : {target_position} 주")
    print(f"이번 거래 수량  : {trade_size:+d} 주")
    print(f"Equity (변경 전/후): {equity_before:.2f} -> {equity_after:.2f}")
    if rolling_win_ratio is not None and dynamic_risk_fraction is not None:
        print(f"rolling_win_ratio={rolling_win_ratio:.3f}, dynamic_risk_fraction={dynamic_risk_fraction:.4f}")
    print("================================")
    print("⚠️ 실제 주문은 자동 실행되지 않습니다. 이 결과를 참고해서 수동으로 주문하세요.")
    print()

    # ----- CSV 로그 저장 -----
    append_live_log_row(
        timestamp=now_utc,
        symbol=SYMBOL,
        price=last_price,
        action=action,
        position_before=position,
        position_after=target_position,
        trade_size=trade_size,
        equity_before=equity_before,
        equity_after=equity_after,
        max_position_size=max_position_size,
        leverage=leverage,
        rolling_win_ratio=rolling_win_ratio,
        dynamic_risk_fraction=dynamic_risk_fraction,
        mode="daily_once",
    )


# ================================
# 엔트리 포인트
# ================================

if __name__ == "__main__":
    try:
        run_live_trading_once()
    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")
