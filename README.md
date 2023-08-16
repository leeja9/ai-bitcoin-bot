# AI/ML Bitcoin Trading Bot

# Description

CS 467 Summer 2023 Capstone Project

Oregon State University

# Team

- Chester Ornes (Industry Mentor)
- James Lee
- Galen Ciszek
- Ravindu Udugampola

# Getting Started

1. Install package requirements
    - `python3 -m venv .venv` (Optional. Create virtual environment)
    - `pip3 install -r requirements.txt`
2. Adjust global parameters in `bot.py`
    - `TRAIN_NEW_MODEL` - True or False
        - A new model will be trained with `N_TIMESTEPS`
    - `DOWNLOAD_NEW_DATA` - True or False
        - Download data from Binance exchange from 2017 ~ present
        - Currently does not work with U.S. ip address (try with vpn)
    - `RENDER_RESULTS` - True or False
        - Render results of tests using `gym-trading-env` renderer
        - Serves with `Flask` on `localhost`
        - Each run of the bot will save the test results in `render_logs` dir
    - `N_TIMESTEPS` - number of training timesteps
    - `SAVED_MODEL` - name of `.zip` file in `saved_models` dir     
        - E.g. `'RecurrentPPO-2023-08-14T02-16-16'`
3. Run bot with `python3 bot.py`

# System Design

![system_design_ideas_v2](https://github.com/leeja9/ai-bitcoin-bot/assets/122495104/e31c0def-02c0-4219-87e1-fed2ad8de5d2)

# Presentation Video

https://github.com/leeja9/ai-bitcoin-bot/assets/122495104/7ad3b4c6-2abc-4378-a2b6-9f703d03e742

# Detailed File Descriptions
- `data/`
    - Default directory for `BitcoinDownloader` and `BitcoinIndicators`
    - Contains cryptocurrency data downloaded from exchange APIs like Binance
- `experimental_multi_env/`
    - Experimental work with multi-vector environments. This can continued in future iterations of this projects to further improve the model.
- `render_logs/`
    - Default directory for rendering test results.
    - `gym_trading_env.renderer` can save and load `.pkl` files and display them via a `Flask` app.
- `saved_models/`
    - Default directory for storing models trained with the `stable_baselines3` package.
    - When loading a model, the `.zip` suffix is not necessary
- `BitcoinDownloader.py`
    - This is a basic wrapper for the `gym_trading_env.download` function
    - Provides function for downloading a default range and parsing the data into `pandas` dataframes
- `BitcoinIndicators.py`
    - Calculates the `features` used in `gym-trading-env` environment
    - The collection of calculated `features` are what's commonly known as the state vector in reinforcement learning.
- `BitcoinMetrics.py`
    - Defines custom metrics for the `gym-trading-env` testing and rendering
- `BitcoinRenderer.py`
    - Can be imported or used as a standalone program for rendering testing data with out performing a new testing run.
- `BitcoinRewards.py`
    -  Defines the reward function that is used in the reinforcement learning.
- `bot.ipynb`
    - A `jupyter` notebook that can be used for testing specific sections of code.
- `bot.py`
    - main program