import pandas as pd
import numpy as np


class Indicators:
    """Calculates technical indicators used in defining market environment
    state from csv data.  Writes to new columns."""

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col="date", parse_dates=True)

        self.lookback_periods = [2, 4, 6, 8, 16, 32, 64]

        # DEPRECATED. Calculating all indicators in all periods listed above.
        # self.roc_periods = [2, 4, 8, 16, 32, 64]
        # self.atr_periods = [2, 4, 8, 16, 32, 64]
        # self.rsi_periods = [2, 4, 8, 16, 32, 64]
        # self.adx_period = [2, 4, 6, 8, 16, 32, 64]

        self.check_date()
        self.calculate_all_indicators()

        # Write all calculation steps to new csv.
        self.data.to_csv("./data/dataset_calculation_columns.csv")

        # Only write results to output csv.
        indicators = ["close", "open", "high", "low", "vol"]
        labels = ["ROC_", "RROC_", "ADX_", "RSI_", "ATR_"]
        for label in labels:
            for period in self.lookback_periods:
                indicator = label + str(period)
                indicators.append(indicator)

        self.data[indicators].to_csv("./data/indicators.csv")

    def check_date(self):
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def calculate_all_indicators(self):
        # Calculate all indicators for all lookback periods.
        for period in self.lookback_periods:
            self.calculate_roc(period)
            self.calculate_rroc(period)
            self.calculate_adx_atr(period)
            self.calculate_rsi(period)

    def calculate_roc(self, period):
        """
        ROC (Rate of Change) - Velocity Measure
        Simple percentage difference between current price and some previous
        price.
        https://www.investopedia.com/terms/r/rateofchange.asp
        """
        # Smooth data with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        self.data[f"roc_ema1_{period}"] = (
            self.data["close"].ewm(span=period, adjust=False).mean()
        )
        self.data[f"roc_ema2_{period}"] = (
            self.data[f"roc_ema1_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"roc_ema3_{period}"] = (
            self.data[f"roc_ema2_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"roc_tema_{period}"] = (
            (3 * self.data[f"roc_ema1_{period}"])
            - (3 * self.data[f"roc_ema2_{period}"])
            + self.data[f"roc_ema3_{period}"]
        )

        # ROC = current_price - previous_price / previous_price
        self.data[f"ROC_{period}"] = (
            self.data[f"roc_tema_{period}"].pct_change(periods=period) * 100
        )

    def calculate_rroc(self, period):
        """
        Rate of Change of ROC - Acceleration Measure
        Simple percentage difference between current ROC and previous ROC.
        """
        # Smooth data with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        self.data[f"rr_ema1_{period}"] = (
            self.data[f"ROC_{period}"].ewm(span=period, adjust=False).mean()
        )
        self.data[f"rr_ema2_{period}"] = (
            self.data[f"rr_ema1_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"rr_ema3_{period}"] = (
            self.data[f"rr_ema2_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"rr_tema_{period}"] = (
            (3 * self.data[f"rr_ema1_{period}"])
            - (3 * self.data[f"rr_ema2_{period}"])
            + self.data[f"rr_ema3_{period}"]
        )

        # ROC = current_price - previous_price / previous_price
        self.data[f"RROC_{period}"] = (
            self.data[f"rr_tema_{period}"].pct_change(periods=period) * 100
        )

    def calculate_adx_atr(self, period):
        """
        ADX (Average Directional Index) - Acceleration Measure
        ATR (Average True Range) - Volatility Measure
        The smoothed average of DX (directional index).
        DX = 100 * abs(pos_DI - neg_DI / pos_DI + neg_DI)
        DI = 100 * smoothed(DM) / smoothed(TR), DM can be +DM or -DM
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        To calculate DM, first calculate UpMove and DownMove
        UpMove = current_high - prev_high
        DownMove = prev_low - curr_low
        If UpMove > DownMove and UpMove > 0, +DM = UpMove, else +DM = 0
        If DownMove > UpMove and DownMove > 0, -DM = DownMove, else -DM = 0
        https://www.investopedia.com/terms/a/adx.asp
        https://en.wikipedia.org/wiki/Average_directional_movement_index?useskin=vector
        """
        # Calculate TR (true range)
        self.data["high-low"] = self.data["high"] - self.data["low"]
        self.data["high-prev_close"] = abs(
            self.data["high"] - self.data["close"].shift(1)
        )
        self.data["low-prev_close"] = abs(
            self.data["low"] - self.data["close"].shift(1)
        )
        self.data[f"TR_{period}"] = self.data[
            ["high-low", "high-prev_close", "low-prev_close"]
        ].max(axis=1)
        self.data[f"ATR_{period}"] = (
            self.data[f"TR_{period}"].rolling(window=period).mean()
        )

        # Calculate DM (directional movement)
        self.data["up_move"] = self.data["high"] - self.data["high"].shift(1)
        self.data["down_move"] = self.data["low"].shift(1) - self.data["low"]
        self.data[f"pos_DM_{period}"] = np.where(
            (self.data["up_move"] > self.data["down_move"])
            & (self.data["up_move"] > 0),
            self.data["up_move"],
            0,
        )
        self.data[f"neg_DM_{period}"] = np.where(
            (self.data["down_move"] > self.data["up_move"])
            & (self.data["down_move"] > 0),
            self.data["down_move"],
            0,
        )

        # Smooth pos_DMwith triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        self.data[f"posdm_ema1_{period}"] = (
            self.data[f"pos_DM_{period}"].ewm(span=period, adjust=False).mean()
        )
        self.data[f"posdm_ema2_{period}"] = (
            self.data[f"posdm_ema1_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"posdm_ema3_{period}"] = (
            self.data[f"posdm_ema2_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"posdm_tema_{period}"] = (
            (3 * self.data[f"posdm_ema1_{period}"])
            - (3 * self.data[f"posdm_ema2_{period}"])
            + self.data[f"posdm_ema3_{period}"]
        )
        self.data[f"smooth_pos_DM_{period}"] = (
            self.data[f"posdm_tema_{period}"].pct_change(
                periods=-(period)) * 100
        )
        # Smooth neg_DM with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        self.data[f"negdm_ema1_{period}"] = (
            self.data[f"neg_DM_{period}"].ewm(span=period, adjust=False).mean()
        )
        self.data[f"negdm_ema2_{period}"] = (
            self.data[f"negdm_ema1_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"negdm_ema3_{period}"] = (
            self.data[f"negdm_ema2_{period}"].ewm(
                span=period, adjust=False).mean()
        )
        self.data[f"negdm_tema_{period}"] = (
            (3 * self.data[f"negdm_ema1_{period}"])
            - (3 * self.data[f"negdm_ema2_{period}"])
            + self.data[f"negdm_ema3_{period}"]
        )
        self.data[f"smooth_neg_DM_{period}"] = (
            self.data[f"neg_DM_{period}"].rolling(window=period).mean()
        )

        # Calculate DI (directional index) positive and negative
        self.data[f"pos_DI_{period}"] = 100 * (
            self.data[f"smooth_pos_DM_{period}"] / self.data[f"ATR_{period}"]
        )
        self.data[f"neg_DI_{period}"] = 100 * (
            self.data[f"smooth_neg_DM_{period}"] / self.data[f"ATR_{period}"]
        )

        # Calculate DX (directional index)
        self.data[f"DX_{period}"] = (
            100
            * (abs(self.data[f"pos_DI_{period}"])
                - abs(self.data[f"neg_DI_{period}"]))
            / (abs(self.data[f"pos_DI_{period}"])
                + abs(self.data[f"neg_DI_{period}"]))
        )

        # Calculate ADX, average of DX over period
        self.data[f"ADX_{period}"] = (
            self.data[f"DX_{period}"].rolling(window=period).mean()
        )

    def calculate_rsi(self, period):
        """
        RSI (Relative Strength Index) - Acceleration and Climax Measure
        Compares the strength of price increases versus the strength of price
        decreases. Used as a measure of whether an asset is overbought or
        oversold.
        https://www.investopedia.com/terms/r/rsi.asp
        https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
        """
        # Calculate daily price change from close the day before.
        self.data["price change"] = self.data["close"].diff()

        # Grab gain and loss values from previous calculation.
        self.data["gain"] = np.where(
            self.data["price change"] > 0, self.data["price change"], 0
        )
        self.data["loss"] = np.where(
            abs(self.data["price change"]) > 0, abs(
                self.data["price change"]), 0
        )

        # Average the gains and losses across lookback period.
        self.data[f"average gain_{period}"] = (
            self.data["gain"].rolling(window=period).mean()
        )
        self.data[f"average loss_{period}"] = (
            self.data["loss"].rolling(window=period).mean()
        )

        # Relative strength
        self.data[f"RS_{period}"] = (
            self.data[f"average gain_{period}"] /
            self.data[f"average loss_{period}"]
        )

        # Relative Strength Index
        self.data[f"RSI_{period}"] = 100 - \
            (100 / (1 + self.data[f"RS_{period}"]))
