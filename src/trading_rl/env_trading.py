import numpy as np
import pandas as pd
import gym
from gym import spaces

from .position_sizing import PositionSizer


class RLTradingEnv(gym.Env):
    """
    강화학습용 트레이딩 환경.
    - 에이전트는 매 스텝마다 행동을 선택:
        0: 포지션 없음(관망)
        1: 롱 진입
        2: 숏 진입
    - 포지션은 한 스텝(다음 캔들) 후 청산된다고 가정 (단순 구조)
    - 포지션 크기와 레버리지는 PositionSizer가 결정
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        price_df: pd.DataFrame,
        sizer: PositionSizer,
        window_size: int = 30,
        initial_equity: float = 100_000.0,
        tick_value: float = 10.0,
    ):
        super(RLTradingEnv, self).__init__()

        assert "close" in price_df.columns, "price_df 에 'close' 컬럼이 필요합니다."
        self.df = price_df.reset_index(drop=True)
        # 항상 1차원 float32 배열로 저장
        self.close = self.df["close"].astype(np.float32).to_numpy().reshape(-1)
        self.window_size = window_size
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.tick_value = float(tick_value)

        self.sizer = sizer
        self.trade_history = []  # 1 / -1 의 이력 (완료된 트레이드 기준)
        self.current_step = 0

        # === 관측값 정의 ===
        # 최근 window_size 개의 수익률 + 현재 equity 비율 + 최근 이동 승률 추정
        obs_dim = window_size + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # === 행동 공간 정의 ===
        # 0: 관망, 1: 롱, 2: 숏
        self.action_space = spaces.Discrete(3)

    def _get_returns(self):
        """
        단순 수익률을 1차원 numpy 배열로 계산.
        """
        prices = self.close.reshape(-1).astype(np.float32)
        rets = np.zeros_like(prices, dtype=np.float32)
        rets[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        return rets

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step

        returns = self.returns[start:end]

        # 혹시라도 2차원으로 들어오면 무조건 1차원으로 펴기
        if returns.ndim > 1:
            returns = returns.reshape(-1)

        # 패딩 (초기 구간)
        if len(returns) < self.window_size:
            pad_len = self.window_size - len(returns)
            pad = np.zeros(pad_len, dtype=np.float32)
            returns = np.concatenate([pad, returns])

        returns = returns.astype(np.float32).reshape(-1)

        # 최근 승률 (간단 추정)
        if len(self.trade_history) == 0:
            rolling_wr = 0.5
        else:
            recent = self.trade_history[-50:]
            wins = sum(1 for t in recent if t == 1)
            rolling_wr = wins / max(1, len(recent))

        equity_ratio = float(self.equity / self.initial_equity)

        # equity_ratio / rolling_wr도 1차원으로
        extra = np.array([equity_ratio, rolling_wr], dtype=np.float32).reshape(-1)

        obs = np.concatenate([returns, extra])
        return obs

    def reset(self):
        self.equity = float(self.initial_equity)
        self.trade_history = []
        # 최소한 window_size 이후부터 시작
        self.current_step = self.window_size

        # 미리 리턴 계산
        self.returns = self._get_returns().astype(np.float32).reshape(-1)

        obs = self._get_observation()
        return obs

    def step(self, action):
        """
        action: 0(관망), 1(롱), 2(숏)
        """
        done = False
        info = {}

        if self.current_step >= len(self.close) - 1:
            done = True
            return self._get_observation(), 0.0, done, info

        price_t = self.close[self.current_step]
        price_tp1 = self.close[self.current_step + 1]
        price_diff = price_tp1 - price_t

        # 포지션 방향
        if action == 0:
            direction = 0  # no position
        elif action == 1:
            direction = 1  # long
        elif action == 2:
            direction = -1  # short
        else:
            raise ValueError("Invalid action.")

        pnl = 0.0

        if direction != 0 and self.equity > 0:
            # 동적 포지션 사이징 + 레버리지
            sizing_info = self.sizer.calculate_size(self.equity, self.trade_history)
            size = sizing_info.get("position_size", 0.0)
            leverage = sizing_info.get("leverage", 1.0)

            # 계약 수 * 레버리지 * 방향 * 가격 변화 * 틱 밸류
            exposure = size * leverage * direction
            pnl = float(exposure * price_diff * self.tick_value)

            # 승/패 기록
            if pnl > 0:
                self.trade_history.append(1)
            elif pnl < 0:
                self.trade_history.append(-1)

            # 자본 업데이트
            self.equity += pnl
            if self.equity <= 0:
                self.equity = 0.0
                done = True

            info.update(sizing_info)
            info["pnl"] = pnl
            info["equity"] = self.equity
            info["action"] = int(action)
        else:
            # 관망 or equity 0
            pnl = 0.0

        # 보상: 초기자본 대비 PnL 비율 (간단 버전)
        reward = pnl / max(self.initial_equity, 1.0)

        # 다음 스텝으로 진행
        self.current_step += 1
        if self.current_step >= len(self.close) - 1:
            done = True

        obs = self._get_observation()

        return obs, float(reward), done, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Equity: {self.equity:.2f}")
