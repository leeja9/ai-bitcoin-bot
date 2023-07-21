import pandas as pd
import numpy as np


class Indicators:
    """Calculates technical indicators used in defining market environment
    state from csv data.  Writes to new columns."""

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, index_col="date", parse_dates=True)

        self.lookback_periods = [2]

        # Deprecated. Calculating all indicators in all periods listed above.
        self.roc_period = 14
        self.atr_period = 14
        self.rsi_period = 14
        self.adx_period = 14

        self.check_date()
        self.calculate_all_indicators()

        # Write all calculation steps to new csv.
        self.data.to_csv("./data/dataset_calculation_columns.csv")

        # Only write results to output csv.
        indicators = []
        labels = ["ROC_", "ADX_", "RSI_", "ATR_"]
        for label in labels:
            for period in self.lookback_periods:
                indicator = label + str(period)
                indicators.append(indicator)

        print(indicators)

        # self.data[indicators].to_csv("./data/indicators.csv")

    def check_date(self):
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def calculate_all_indicators(self):
        # Calculate all indicators for all lookback periods.
        for period in self.lookback_periods:
            self.calculate_roc(period)
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
        self.data["ema1"] = self.data["close"].ewm(
            span=period, adjust=False).mean()
        self.data["ema2"] = self.data["ema1"].ewm(
            span=period, adjust=False).mean()
        self.data["ema3"] = self.data["ema2"].ewm(
            span=period, adjust=False).mean()
        self.data["tema"] = (
            (3 * self.data["ema1"]) -
            (3 * self.data["ema2"]) + self.data["ema3"]
        )

        # ROC = current_price - previous_price / previous_price
        self.data[f"ROC_{period}"] = (
            self.data["tema"].pct_change(periods=-(period)) * 100
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
        # Grab initial values from data.
        high = self.data['high']
        low = self.data['low']
        prev_close = self.data['close'].shift(-1)
        prev_high = self.data['high'].shift(-1)
        prev_low = self.data['low'].shift(-1)

        # Calculate TR and ATR
        high-low = high - low
        high-prev_close = high - prev_close
        low-prev_close = low - prev_close

        self.data['TR'] = max(high-low, high-prev_close, low-prev_close)
        self.data[f'ATR_{period}'] = self.data['TR'].rolling(window=period).mean()

        # Calculate DMs
        up_move = high - prev_high
        down_move = prev_low - low

        pos_DM = 



























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
        self.rsi_period = 14
        self.data["average gain"] = self.data["gain"].rolling(
            window=period).mean()
        self.data["average loss"] = self.data["loss"].rolling(
            window=period).mean()

        # Relative strength
        self.data["RS"] = self.data["average gain"] / self.data["average loss"]

        # Relative Strength Index
        self.data[f"RSI_{period}"] = 100 - (100 / (1 + self.data["RS"]))
