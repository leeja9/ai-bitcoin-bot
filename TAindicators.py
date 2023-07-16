import pandas as pd
import numpy as np


class Indicators:
    """Calculates technical indicators used in defining market environment
    state from csv data.  Writes to new columns."""

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col="date", parse_dates=True)

        self.roc_period = 14
        self.atr_period = 14
        self.rsi_period = 14

        self.check_date()
        self.calculate_all_indicators()

        self.data.to_csv(csv_file)

    def check_date(self):
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def calculate_all_indicators(self):
        self.calculate_roc()
        self.calculate_adx()
        self.calculate_rsi()
        self.calculate_atr()

    def set_roc_period(self, value):
        self.roc_period = value

    def set_atr_period(self, value):
        self.atr_period = value

    def set_rsi_period(self, value):
        self.rsi_period = value

    # Get methods to return indicators.
    # TODO: consult with team to see how these should be best implemented.

    def get_roc2(self):
        return self.data["ROC2"]

    def get_roc4(self):
        return self.data["ROC4"]

    def get_roc8(self):
        return self.data["ROC8"]

    def get_roc16(self):
        return self.data["ROC16"]

    def get_roc32(self):
        return self.data["ROC32"]

    def calculate_roc(self):
        """
        ROC (Rate of Change) - Velocity Measure
        Simple percentage difference between current price and some previous
        price.
        https://www.investopedia.com/terms/r/rateofchange.asp
        """
        # Smooth data with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        self.data["ema1"] = (
            self.data["close"].ewm(span=self.roc_period, adjust=False).mean()
        )
        self.data["ema2"] = (
            self.data["ema1"].ewm(span=self.roc_period, adjust=False).mean()
        )
        self.data["ema3"] = (
            self.data["ema2"].ewm(span=self.roc_period, adjust=False).mean()
        )
        self.data["tema"] = (
            (3 * self.data["ema1"]) -
            (3 * self.data["ema2"]) + self.data["ema3"]
        )

        # ROC = current_price - previous_price / previous_price
        self.data["ROC2"] = self.data["tema"].pct_change(periods=-2) * 100
        self.data["ROC4"] = self.data["tema"].pct_change(periods=-4) * 100
        self.data["ROC8"] = self.data["tema"].pct_change(periods=-8) * 100
        self.data["ROC16"] = self.data["tema"].pct_change(periods=-16) * 100
        self.data["ROC32"] = self.data["tema"].pct_change(periods=-32) * 100

    def calculate_adx(self):
        """
        ADX (Average Directional Index) - Acceleration Measure
        The smoothed average of DX (directional index).
        DX = 100 * abs(+DI - -DI / +DI + -DI)
        DI = 100 * smoothed(DM) / smoothed(TR), DM can be +DM or -DM
        TR =
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
            self.data["high"] - self.data["close"].shift(-1)
        )
        self.data["low-prev_close"] = abs(
            self.data["low"] - self.data["close"].shift(-1)
        )
        self.data["TR"] = self.data[
            ["high-low", "high-prev_close", "low-prev_close"]
        ].max(axis=1)

        # Calculate DM (directional movement)
        self.data["up_move"] = self.data["high"] - self.data["high"].shift(-1)
        self.data["down_move"] = self.data["low"].shift(-1) - self.data["low"]
        self.data["pos_DM"] = np.where(
            (self.data["up_move"] > self.data["down_move"])
            & (self.data["up_move"] > 0),
            self.data["up_move"],
            0,
        )
        self.data["neg_DM"] = np.where(
            (self.data["down_move"] > self.data["up_move"])
            & (self.data["down_move"] > 0),
            self.data["down_move"],
            0,
        )

    def calculate_rsi(self):
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

        # Average the gains and losses across 14 days.
        self.rsi_period = 14
        self.data["average gain"] = (
            self.data["gain"].rolling(window=self.rsi_period).mean()
        )
        self.data["average loss"] = (
            self.data["loss"].rolling(window=self.rsi_period).mean()
        )

        # Relative strength
        self.data["RS"] = self.data["average gain"] / self.data["average loss"]

        # Relative Strength Index
        self.data["RSI"] = 100 - (100 / (1 + self.data["RS"]))

    def calculate_atr(self):
        """
        ATR (Average True Range) - Volatility Measure
        Just the 14-period average of the True Range calculation previously.
        """
        self.data["average TR"] = self.data["TR"].rolling(
            window=self.atr_period).mean()
