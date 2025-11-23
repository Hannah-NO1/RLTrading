import numpy as np
from abc import ABC, abstractmethod


class PositionSizer(ABC):
    """
    포지션 사이저의 추상 기본 클래스.
    모든 구체적인 사이저 클래스는 이 클래스를 상속받아야 합니다.
    """

    @abstractmethod
    def calculate_size(self, equity, trade_history):
        """
        현재 자본과 거래 내역을 기반으로 포지션 크기를 계산합니다.

        :param equity: 현재 총 자본 (float)
        :param trade_history: 거래 결과 리스트 (1: 승, -1: 패)
        :return: 포지션 크기 및 관련 정보 딕셔너리
        """
        pass


class FixedFractionalSizer(PositionSizer):
    """
    고정 비율 리스크 모델
    """

    def __init__(self, risk_fraction=0.02, stop_loss_pips=50, pip_value=10):
        """
        :param risk_fraction: 자본 대비 리스크 비율 (예: 0.02 = 2%)
        :param stop_loss_pips: 1계약당 손절 폭(핍 또는 가격 단위)
        :param pip_value: 핍 가치(1핍당 금액)
        """
        self.risk_fraction = risk_fraction
        self.stop_loss_pips = stop_loss_pips
        self.pip_value = pip_value

    def calculate_size(self, equity, trade_history):
        risk_amount = equity * self.risk_fraction
        risk_per_contract = self.stop_loss_pips * self.pip_value
        if risk_per_contract <= 0:
            return {
                "position_size": 0,
                "leverage": 1.0,
                "rolling_win_ratio": None,
                "dynamic_risk_fraction": self.risk_fraction,
            }

        position_size = int(np.floor(risk_amount / risk_per_contract))
        return {
            "position_size": position_size,
            "leverage": 1.0,
            "rolling_win_ratio": None,
            "dynamic_risk_fraction": self.risk_fraction,
        }


class WinRatioModulatedSizer(PositionSizer):
    """
    이동 승률과 시그모이드 함수를 사용하여
    리스크와 레버리지를 동적으로 조절하는 사이저.

    보고서의 4.3, 5.2 내용을 구현한 클래스.
    """

    def __init__(
        self,
        lookback_period=60,
        base_fraction=0.02,
        sigmoid_L=2.0,
        sigmoid_k=10,
        sigmoid_wr0=0.35,
        use_dynamic_leverage=True,
        max_leverage=2.0,
        stop_loss_pips=50,
        pip_value=10,
    ):
        """
        :param lookback_period: 이동 승률 계산에 사용할 최근 거래 수
        :param base_fraction: 기본 리스크 비율 (예: 0.02 = 2%)
        :param sigmoid_L: 시그모이드 상한(최대 리스크 승수)
        :param sigmoid_k: 시그모이드 기울기 (민감도)
        :param sigmoid_wr0: 기준 승률 (이 값 근처에서 가장 민감하게 반응)
        :param use_dynamic_leverage: 동적 레버리지 사용 여부
        :param max_leverage: 최대 레버리지 배수
        :param stop_loss_pips: 1계약당 손절 폭(핍 또는 가격 단위)
        :param pip_value: 핍 가치(1핍당 금액)
        """
        self.lookback_period = lookback_period
        self.base_fraction = base_fraction
        self.sigmoid_L = sigmoid_L
        self.sigmoid_k = sigmoid_k
        self.sigmoid_wr0 = sigmoid_wr0
        self.use_dynamic_leverage = use_dynamic_leverage
        self.max_leverage = max_leverage
        self.stop_loss_pips = stop_loss_pips
        self.pip_value = pip_value

    def _sigmoid(self, x: float) -> float:
        """
        승률 x (0~1)를 받아 리스크 승수(0.5배 ~ L배)를 반환하는 시그모이드 함수.
        """
        if self.sigmoid_L <= 0:
            return 1.0

        # 기본 리스크 비율을 0.5배에서 L배까지 조절하도록 스케일링
        return (self.sigmoid_L - 0.5) / (
            1.0 + np.exp(-self.sigmoid_k * (x - self.sigmoid_wr0))
        ) + 0.5

    def _compute_rolling_win_ratio(self, trade_history):
        """
        trade_history (1/-1 리스트)에서 최근 lookback_period만 보고 승률 계산.
        """
        if not trade_history:
            return 0.5

        recent = trade_history[-self.lookback_period :]
        if len(recent) == 0:
            return 0.5

        wins = sum(1 for t in recent if t == 1)
        return wins / len(recent)

    def calculate_size(self, equity, trade_history):
        """
        equity와 최근 승/패 기록을 이용하여
        - 동적 리스크 비율
        - 레버리지
        - 포지션 크기 (계약 수, or 주 수 상한)
        를 계산.
        """

        rolling_win_ratio = self._compute_rolling_win_ratio(trade_history)

        # 1. 신뢰도 점수 (리스크 조절 계수)
        confidence_score = self._sigmoid(rolling_win_ratio)

        # 2. 동적 리스크 비율
        dynamic_risk_fraction = self.base_fraction * confidence_score

        # 3. 레버리지 계산
        leverage = 1.0
        if self.use_dynamic_leverage:
            # confidence_score를 0~1 범위로 정규화
            normalized_confidence = (confidence_score - 0.5) / (
                self.sigmoid_L - 0.5
            )
            normalized_confidence = max(0.0, min(1.0, normalized_confidence))
            # 매우 높은 신뢰도에서만 레버리지가 크게 증가하도록 제곱 사용
            leverage = 1.0 + (self.max_leverage - 1.0) * (normalized_confidence**2)

        # 4. 최종 포지션 크기 (계약 수)
        risk_amount = equity * dynamic_risk_fraction
        risk_per_contract = self.stop_loss_pips * self.pip_value
        if risk_per_contract <= 0:
            position_size = 0
        else:
            position_size = int(np.floor(risk_amount / risk_per_contract))

        return {
            "position_size": position_size,
            "leverage": float(leverage),
            "rolling_win_ratio": float(rolling_win_ratio),
            "dynamic_risk_fraction": float(dynamic_risk_fraction),
        }
