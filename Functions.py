import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy.stats import skew, kurtosis, kstest, norm
from statsmodels.tsa.stattools import adfuller
import ast


date_threshold = datetime(2017, 8, 21)

crypto_data = {}
model_data = {}
google_data = {}
crypto_symbols = []

def fetch_and_save_crypto_data(crypto_symbol, filename):
    start_date = datetime(2017, 7, 1)
    end_date = datetime(2024, 1, 1)
    url = f'https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart/range?vs_currency=usd&from={start_date.timestamp()}&to={end_date.timestamp()}'
    response = requests.get(url)
    data = response.json()

    price_data = pd.DataFrame(data['prices'], columns=['time', 'price'])
    volume_data = pd.DataFrame(data['total_volumes'], columns=['time', 'volume'])

    price_data['time'] = pd.to_datetime(price_data['time'], unit='ms')
    volume_data['time'] = pd.to_datetime(volume_data['time'], unit='ms')

    merged_data = pd.merge(price_data, volume_data, on='time', how='inner')

    merged_data.to_csv(filename, index=False)
    return merged_data

# Code for Retrieving New Data
for symbol in crypto_symbols:
    filename = f'{symbol}_data.csv'
    crypto_data[symbol] = fetch_and_save_crypto_data(symbol, filename)

def read_saved_crypto_data(symbols):
    for symbol in symbols:
        filename = f'{symbol}_data.csv'
        dati = pd.read_csv(filename)
        dati['time'] = pd.to_datetime(dati['time'])
        dati.set_index('time', inplace=True)
        crypto_data[symbol] = dati
    return crypto_data

def read_model_data(symbols):
    for symbol in symbols:
        filename = f'{symbol}1_data.csv'
        dati = pd.read_csv(filename)
        dati.set_index('time', inplace=True)
        model_data[symbol] = dati
    return model_data

def read_google_data(symbols):
    for symbol in symbols:
        filename = f'{symbol}_searches.csv'
        dati = pd.read_csv(filename)
        dati.set_index('Settimana', inplace=True)
        google_data[symbol] = dati
    return google_data

def read_cppi_data(symbols):
    cppi_data = {}
    for symbol in symbols:
        filename = f'{symbol}_cppi.csv'
        dati = pd.read_csv(filename, index_col='time')
        dati.index = pd.to_datetime(dati.index)
        cppi_data[symbol] = dati
    return cppi_data


def extract_log_data(loaded_data, date_threshold):
    log_data = {}

    for symbol, data in loaded_data.items():

        price_data = data[['price']].copy()
        volume_data = data[['volume']].copy()

        price_data['log_return'] = np.log(price_data['price']) - np.log(price_data['price'].shift(1))
        volume_data['log_volume_variation'] = np.log(volume_data['volume']) - np.log(volume_data['volume'].shift(1))

        merged_data = pd.concat([data['price'], data['volume'], price_data['log_return'], volume_data['log_volume_variation']], axis=1)
        merged_data.index = pd.to_datetime(data.index)

        merged_data = merged_data[merged_data.index >= date_threshold].copy()
        merged_data.reset_index(inplace=True)

        merged_data['time'] = pd.to_datetime(merged_data['time'])
        merged_data.set_index('time', inplace=True)

        log_data[symbol] = merged_data

    return log_data

def test_normality(selected_log_data):
    normality_results_dict = {}

    log_return = selected_log_data['log_return']
    log_volume_variation = selected_log_data['log_volume_variation']

    # Log Returns Analysis
    ks_stat_return, ks_pval_return = kstest(log_return, 'norm')
    ks_normality_result_return = ks_pval_return >= 0.5
    normality_results_dict[
        'KS Test on Returns'] = 'Normal' if ks_normality_result_return else 'Not Normally Distributed'
    normality_results_dict['Returns Skewness'] = skew(log_return)
    normality_results_dict['Returns Kurtosis'] = kurtosis(log_return)

    # Log Volumes Analysis
    ks_stat_volume, ks_pval_volume = kstest(log_volume_variation, 'norm')
    ks_normality_result_volume = ks_pval_volume >= 0.5
    normality_results_dict[
        'KS Test on Volumes'] = 'Normal' if ks_normality_result_volume else 'Not Normally Distributed'
    normality_results_dict['Volumes Skewness'] = skew(log_volume_variation)
    normality_results_dict['Volumes Kurtosis'] = kurtosis(log_volume_variation)

    normality_results_df = pd.DataFrame([normality_results_dict])

    return normality_results_df

def calculate_returns_volatility(selected_price_data):

    first_value = selected_price_data.iloc[0]

    weekly_data = selected_price_data.resample('W-Mon').last()
    monthly_data = selected_price_data.resample('M').last()
    yearly_data = selected_price_data.resample('A').last()

    monthly_data.loc[date_threshold] = first_value
    yearly_data.loc[date_threshold] = first_value
    monthly_data.sort_index(inplace=True)
    yearly_data.sort_index(inplace=True)

    weekly_returns = np.log(weekly_data / weekly_data.shift(1))
    monthly_returns = np.log(monthly_data / monthly_data.shift(1))
    yearly_returns = np.log(yearly_data / yearly_data.shift(1))

    mean_returns = {
        'Weekly': weekly_returns.mean(),
        'Monthly': monthly_returns.mean(),
        'Yearly': yearly_returns.mean()
    }

    volatility = {
        'Weekly': np.std(weekly_returns) * np.sqrt(52),
        'Monthly': np.std(monthly_returns) * np.sqrt(12),
        'Yearly': np.std(yearly_returns)
    }

    returns_volatility_data = {
        'Mean Returns (Weekly)': f'{mean_returns["Weekly"] * 100:.2f} %',
        'Realized Volatility (Weekly)': f'{volatility["Weekly"] * 100:.2f} %',
        'Mean Returns (Monthly)': f'{mean_returns["Monthly"] * 100:.2f} %',
        'Realized Volatility (Monthly)': f'{volatility["Monthly"] * 100:.2f} %',
        'Mean Returns (Yearly)': f'{mean_returns["Yearly"] * 100:.2f} %',
        'Realized Volatility (Yearly)': f'{volatility["Yearly"] * 100:.2f} %'
    }

    df_results = pd.DataFrame([returns_volatility_data])

    return df_results

###################################################

def create_state_statistics(selected_model_data):
    state_statistics = []

    total_length = len(selected_model_data)

    for state in sorted(selected_model_data['Decoded_States'].unique()):
        state_data = selected_model_data[selected_model_data['Decoded_States'] == state]
        frequency = len(state_data) / total_length * 100
        average_log_return = state_data['log_return'].mean() * 100
        average_volume = state_data['log_volume_variation'].mean() * 100

        state_statistics.append({
            'Hidden States': f'State {state+1}',
            'Frequency (%)': f'{frequency:.2f} %',
            'Avg Log Return (%)': f'{average_log_return:.2f} %',
            'Avg Log Volume (%)': f'{average_volume:.2f} %'
        })

    state_statistics_df = pd.DataFrame(state_statistics)

    return state_statistics_df


def calculate_adf(selected_price_data):
    # Schwert Rule for Lags
    T = len(selected_price_data)
    lags = int(np.ceil(12 * (T / 100) ** (1/4)))

    adf_result = adfuller(selected_price_data, maxlag=lags)

    adf_data = {
        'P-Value': round(adf_result[1], 3),
        'T-Stat': round(adf_result[0], 3),
        'Critical Values': {key: round(val, 3) for key, val in adf_result[4].items()}
    }

    def evaluate_stationarity(test_statistic, critical_value):
        return "Stationary" if test_statistic < critical_value else "Not Stationary"

    adf_results = pd.DataFrame(columns=['P-Value', 'T-Stat', 'T-Stat 1%', 'T-Stat 5%', 'T-Stat 10%', 'Result 1%', 'Result 5%', 'Result 10%'])

    adf_results.loc['ADF'] = [
        adf_data['P-Value'],
        adf_data['T-Stat'],
        adf_data['Critical Values']['1%'],
        adf_data['Critical Values']['5%'],
        adf_data['Critical Values']['10%'],
        evaluate_stationarity(adf_data['T-Stat'], adf_data['Critical Values']['1%']),
        evaluate_stationarity(adf_data['T-Stat'], adf_data['Critical Values']['5%']),
        evaluate_stationarity(adf_data['T-Stat'], adf_data['Critical Values']['10%'])
    ]

    return adf_results


def calculate_log_likelihoods(selected_model_data, selected_chmm):

    means = ast.literal_eval(selected_chmm['Means'])
    covariances = ast.literal_eval(selected_chmm['Covariances'])
    num_states = len(means)

    nomodel_return = selected_model_data['log_return']
    nomodel_volume = selected_model_data['log_volume_variation']
    nomodel_sample = len(nomodel_return)

    nomodel_return_ll = np.sum(norm.logpdf(nomodel_return, np.mean(nomodel_return), np.std(nomodel_return)))
    nomodel_volume_ll = np.sum(norm.logpdf(nomodel_volume, np.mean(nomodel_volume), np.std(nomodel_volume)))

    returns_ll = [nomodel_return_ll]
    volumes_ll = [nomodel_volume_ll]
    samples = [nomodel_sample]

    for i in range(num_states):
        state_data = selected_model_data[selected_model_data['Decoded_States'] == i]
        sample_size = len(state_data)
        log_return = state_data['log_return']
        log_volume = state_data['log_volume_variation']

        mean_log_return, cov_log_return = means[i][0], covariances[i][0][0]
        mean_log_volume, cov_log_volume = means[i][1], covariances[i][1][1]

        returns_observed_ll = np.sum(norm.logpdf(log_return, mean_log_return, np.sqrt(cov_log_return)))
        volumes_observed_ll = np.sum(norm.logpdf(log_volume, mean_log_volume, np.sqrt(cov_log_volume)))

        returns_ll.append(returns_observed_ll)
        volumes_ll.append(volumes_observed_ll)
        samples.append(sample_size)

    returns_ll.append(selected_chmm['LogLikelihood'])
    volumes_ll.append(selected_chmm['LogLikelihood'])
    samples.append(nomodel_sample)

    returns_ll = [round(val, 2) for val in returns_ll]
    volumes_ll = [round(val, 2) for val in volumes_ll]

    columns = [''] + ['No Model'] + [f'Hidden State {i+1}' for i in range(num_states)] + ['HMM']
    index = ['Log Returns', 'Log Volumes', 'Sample']
    data = [[index[0]] + returns_ll, [index[1]] + volumes_ll, [index[2]] + samples]

    return pd.DataFrame(data, index=index, columns=columns)

def create_summary_table(selected_cppi):
    summary_data = {}

    final_values = {key: selected_cppi[key].iloc[-1] for key in ['Backup', 'CPPI', 'Strategy', 'HOLD', 'SPREAD']}
    summary_data = final_values

    summary_table = pd.DataFrame(summary_data, index=[0])

    return summary_table


def create_multiplier_table():
    data = [
        ['< -0.2', '-0.2 to < 0.1','0.1 to < 0.2','0.2 to < 0.4','>= 0.4'],
        [1.1, 1.4, 1.9, 1.6, 2]
    ]
    index = ['Multiplier', 'Value']
    multiplier_table = pd.DataFrame(data, index=index)
    multiplier_table.columns = ['Range 1', 'Range 2', 'Range 3', 'Range 4', 'Range 5']
    return multiplier_table

def calculate_expected_value_for_state(transmat, means):
    expected_values = np.sum(transmat * means, axis=1) * 100
    return expected_values

def get_multiplier(expected_value):
    if expected_value < -0.2:
        return 1.1
    elif -0.2 <= expected_value < 0.1:
        return 1.4
    elif 0.1 <= expected_value < 0.2:
        return 1.9
    elif 0.2 <= expected_value < 0.4:
        return 1.6
    else:
        return 2

def create_state_multiplier_table(transmat, means):
    expected_values = calculate_expected_value_for_state(transmat, means)
    multipliers = [get_multiplier(value) for value in expected_values]
    formatted_expected_values = [f'{value:.3f} %' for value in expected_values]

    data = {
        'States': [f'Hidden State {i+1}' for i in range(len(expected_values))],
        'Expected Values': formatted_expected_values,
        'Multipliers': multipliers
    }

    return pd.DataFrame(data)

