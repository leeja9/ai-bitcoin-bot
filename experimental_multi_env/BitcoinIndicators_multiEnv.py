import pandas as pd
import numpy as np
from pathlib import Path


class Indicators:
    """Calculates technical indicators used in defining market environment
    state from csv data.  Writes to new columns."""

    def __init__(self, data_path: str, index_col: str = "date_open"):
        self.index_col = index_col
        if not Path(data_path).is_file():
            raise FileNotFoundError(data_path)
        elif data_path.split('.')[-1] == 'csv':
            self.data = pd.read_csv(data_path)
        elif data_path.split('.')[-1] == 'pkl':
            self.data = pd.read_pickle(data_path)
        else:
            raise NameError("'{}' is not a valid filename. must be '.pkl' or '.csv'".format(data_path))

        self.lookback_periods = [2, 4, 6, 8, 16, 32, 64]

        # DEPRECATED. Calculating all indicators in all periods listed above.
        # self.roc_periods = [2, 4, 8, 16, 32, 64]
        # self.atr_periods = [2, 4, 8, 16, 32, 64]
        # self.rsi_periods = [2, 4, 8, 16, 32, 64]
        # self.adx_period = [2, 4, 6, 8, 16, 32, 64]

        self.check_date()
        self.calculate_all_indicators()

    def to_csv(self, data_path: str):
        self.data.to_csv(f"{data_path}")

    def to_pickle(self, data_path: str):
        self.data.to_pickle(f"{data_path}")

    def check_date(self):
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)

    def calculate_all_indicators(self):
        # Calculate all indicators for all lookback periods.
        self.data.reset_index(inplace=True)
        for period in self.lookback_periods:
            self.calculate_roc(period)
            self.calculate_rroc(period)
            self.calculate_adx_atr(period)
            self.calculate_rsi(period)
        self.data[self.index_col] = pd.to_datetime(self.data[self.index_col])
        self.data.set_index(self.index_col, inplace=True)

    def calculate_roc(self, period):
        """
        ROC (Rate of Change) - Velocity Measure
        Simple percentage difference between current price and some previous
        price.
        https://www.investopedia.com/terms/r/rateofchange.asp
        """
        roc_ema1 = self.data["close"].ewm(span=period, adjust=False).mean()
        roc_ema2 = roc_ema1.ewm(span=period, adjust=False).mean()
        roc_ema3 = roc_ema2.ewm(span=period, adjust=False).mean()
        roc_tema = 3 * roc_ema1 - 3 * roc_ema2 + roc_ema3

        # ROC = current_price - previous_price / previous_price
        feature_ROC = roc_tema.pct_change(periods=period) * 100
        self.data = pd.concat(
            (self.data, pd.DataFrame({f"feature_ROC_{period}": feature_ROC})),
            axis=1
        )

    def calculate_rroc(self, period):
        """
        Rate of Change of ROC - Acceleration Measure
        Simple percentage difference between current ROC and previous ROC.
        """
        # Smooth data with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        rr_ema1 = self.data[f"feature_ROC_{period}"].ewm(span=period, adjust=False).mean()
        rr_ema2 = rr_ema1.ewm(span=period, adjust=False).mean()
        rr_ema3 = rr_ema2.ewm(span=period, adjust=False).mean()
        rr_tema = 3 * rr_ema1 - 3 * rr_ema2 + rr_ema3
        # ROC = current_price - previous_price / previous_price
        feature_RROC = rr_tema.pct_change(periods=period) * 100
        self.data = self.data.join(pd.DataFrame({f"feature_RROC_{period}": feature_RROC}))

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
        high_low = self.data["high"] - self.data["low"]
        high_prev_close = abs(self.data["high"] - self.data["close"].shift(1))
        low_prev_close = abs(self.data["low"] - self.data["close"].shift(1))
        high_low_data = pd.DataFrame({"high-low": high_low,
                                      "high-prev_close": high_prev_close,
                                      "low-prev_close": low_prev_close})
        TR = high_low_data[["high-low", "high-prev_close", "low-prev_close"]].max(axis=1)
        feature_ATR = TR.rolling(window=period).mean()

        # Calculate DM (directional movement)
        up_move = self.data["high"] - self.data["high"].shift(1)
        down_move = self.data["low"].shift(1) - self.data["low"]
        DM = pd.DataFrame({'up_move': up_move, 'down_move': down_move})
        pos_DM = pd.Series(np.where(
            (DM["up_move"] > DM["down_move"]) & (DM["up_move"] > 0),
            DM["up_move"],
            0
        ))
        neg_DM = pd.Series(np.where(
            (DM["down_move"] > DM["up_move"]) & (DM["down_move"] > 0),
            DM["down_move"],
            0
        ))
        # Smooth pos_DMwith triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3

        posdm_ema1 = pos_DM.ewm(span=period, adjust=False).mean()
        posdm_ema2 = posdm_ema1.ewm(span=period, adjust=False).mean()
        posdm_ema3 = posdm_ema2.ewm(span=period, adjust=False).mean()
        posdm_tema = (3 * posdm_ema1) - (3 * posdm_ema2) + posdm_ema3
        smooth_pos_DM = posdm_tema.pct_change(periods=-(period)) * 100

        # Smooth neg_DM with triple exponential moving average (TEMA)
        # TEMA = 3(ema1) - 3(ema2) + ema3
        negdm_ema1 = neg_DM.ewm(span=period, adjust=False).mean()
        negdm_ema2 = negdm_ema1.ewm(span=period, adjust=False).mean()
        negdm_ema3 = negdm_ema2.ewm(span=period, adjust=False).mean()
        negdm_tema = (3 * negdm_ema1) - (3 * negdm_ema2) + negdm_ema3
        smooth_neg_DM = negdm_tema.rolling(window=period).mean()

        # Calculate DI (directional index) positive and negative
        pos_DI = 100 * (smooth_pos_DM / feature_ATR)
        neg_DI = 100 * (smooth_neg_DM / feature_ATR)

        # Calculate DX (directional index)
        # DX = 100 * (pos_DI.abs() - neg_DI.abs()) / (pos_DI.abs() + neg_DI.abs())
        DX = 100 * (abs(pos_DI) - abs(neg_DI)) / (abs(pos_DI) + abs(neg_DI))
        # Calculate ADX, average of DX over period
        feature_ADX = DX.rolling(window=period).mean()
        self.data = self.data.join(pd.DataFrame({
                                   f"feature_ATR_{period}": feature_ATR,
                                   f"feature_ADX_{period}": feature_ADX}))

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
        price_change = self.data["close"].diff()
        self.data["price change"] = self.data["close"].diff()

        # Grab gain and loss values from previous calculation.
        gain = price_change.where(price_change > 0, 0)
        loss = price_change.abs().where(price_change.abs() > 0, 0)

        # Average the gains and losses across lookback period.
        average_gain = gain.rolling(window=period).mean()
        average_loss = loss.rolling(window=period).mean()

        # Relative strength
        RS = average_gain / average_loss

        # Relative Strength Index
        feature_RSI = 100 - (100 / (1 + RS))
        self.data = self.data.join(pd.DataFrame({f"feature_RSI_{period}": feature_RSI}))


if __name__ == "__main__":
    from BitcoinDownloader import get_dataframes
    training, testing_df = get_dataframes()
    data_path = './data/banana.csv'
    from ta_indicators import Indicators as Indicators2
    import warnings

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df0 = testing_df
    ind_path = './data/indicators.csv'
    df0.to_csv(ind_path)
    indicators = Indicators2(ind_path)
    df0 = pd.read_csv(ind_path)
    df0["date_open"] = pd.to_datetime(df0["date_open"])
    df0.set_index("date_open", inplace=True)
    df0.dropna(inplace=True)

    testing_df.to_csv(ind_path)
    indicators = Indicators(ind_path)
    indicators.to_csv(ind_path)
    testing_df = pd.read_csv(ind_path)
    testing_df["date_open"] = pd.to_datetime(testing_df["date_open"])
    testing_df.set_index("date_open", inplace=True)
    testing_df.dropna(inplace=True)

    for col in testing_df.columns:
        if col in df0.columns:
            isEqual = df0[col].equals(testing_df[col])
            if not isEqual:
                print(f"{col}: {isEqual}")
                for i in range(len(df0)):
                    if df0[col][i] != testing_df[col][i]:
                        print(f"{df0[col][i]} != {testing_df[col][i]} at index {i}")
                        break
