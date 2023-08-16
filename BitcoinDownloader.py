import gym_trading_env
from gym_trading_env.downloader import download
from datetime import datetime
from ccxt.base.errors import ExchangeNotAvailable
from pathlib import Path
import pandas as pd
import os


def download_exchange_data(since: datetime = datetime.fromisoformat("2017-01-01"),
                           until: datetime = datetime.today(),
                           dir: str = 'data',
                           exchange_names: list[str] = ['binance'],
                           symbols: list[str] = ['BTC/USDT'],
                           timeframe: str = "1h"):
    """Download exchange data

    Args:
        since (datetime, optional): Defaults to datetime.fromisoformat("2017-01-01").
        until (datetime, optional): Defaults to datetime.today().
        dir (str, optional): Directory path. Defaults to 'data'.
        exchange_names (list[str], optional): Defaults to ['binance'].
        symbols (list[str], optional): Defaults to ['BTC/USDT'].
        timeframe (str, optional): Defaults to "1h".
    """
    print("Starting downloads...")
    try:
        for exchange in exchange_names:
            for symbol in symbols:
                download_path = './{}/{}-{}-{}.pkl'.format(
                    dir,
                    exchange,
                    ''.join(symbol.split('/')),
                    timeframe
                )
                if Path(download_path).is_file():
                    redownload = input("'{}' already exists. Download again? (y/n): ".format(download_path))
                    if redownload.lower() not in ['y', 'yes', 'ye', 'yeah', 'sure', 'ok']:
                        print("Skipping {}-{}".format(exchange, symbol))
                        break
                print("Downloading {} data from {} exchange to path: {}".format(
                    symbol,
                    exchange,
                    download_path
                ))
                download(
                    exchange_names=exchange_names,
                    symbols=symbols,
                    timeframe=timeframe,
                    dir=dir,
                    since=since,
                    until=until
                )
        print("Finished all downloads")
    except ExchangeNotAvailable as ex:
        print("The exchange API doesn't allow downloads from ip addresses in your country")
        print("Try downloading again after connecting to a vpn in a different country")
        print()
        print(ex)


def _import_data(data_path: str) -> pd.DataFrame:
    if Path(data_path).is_file():
        if data_path.split('.')[-1] == 'pkl':
            return pd.read_pickle(data_path)
        if data_path.split('.')[-1] == 'csv':
            return pd.read_csv(data_path)
        else:
            raise NameError('{} invalid. provide path to .csv or .pkl'.format(data_path))
    else:
        raise FileNotFoundError('{} file not found'.format('data_path'))


def get_dataframes(data_path: str = './data/binance-BTCUSDT-1h.pkl') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get training and testing dataframes

    Args:
        data_path (str, optional): Defaults to './data/binance-BTCUSDT-1h.pkl'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: training dataframe, testing dataframe
    """
    df = _import_data(data_path)
    # df.reset_index(inplace=True)
    # df.rename(columns={'date_open': 'date', 'volume': 'vol'}, inplace=True)
    # df['date'] = pd.to_datetime(df['date'])
    # df.set_index('date', inplace=True)
    truncate_date = pd.to_datetime("2023-01-01")
    training_dataframe = df.truncate(after=truncate_date)
    testing_dataframe = df.truncate(before=truncate_date)
    return training_dataframe, testing_dataframe


if __name__ == "__main__)":
    data_dir = './data/'
    curr_time = datetime.isoformat(datetime.today()).split(".")[0].replace(":", "-")
    data_name = f'binance-BTCUSDT-1h-{curr_time}.pkl'
    data_path = data_dir + data_name
    os.makedirs(data_dir, exist_ok=True)
    download_exchange_data()
    df = pd.read_pickle(data_name)
    df.to_csv(data_dir + data_name.split(".")[0] + ".csv")
