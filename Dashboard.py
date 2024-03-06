from Functions import read_model_data
from Functions import read_google_data
from Functions import read_cppi_data
from Functions import test_normality
from Functions import calculate_returns_volatility
from Functions import create_state_statistics
from Functions import calculate_adf
from Functions import calculate_log_likelihoods

from Functions import create_summary_table
from Functions import calculate_expected_value_for_state
from Functions import get_multiplier
from Functions import create_state_multiplier_table

from Plots import plot_price_chart
from Plots import plot_searches_chart
from Plots import plot_density_figure
from Plots import plot_qq_figure

from Plots import states_plot
from Plots import plot_heatmap
from Plots import plot_log_returns_distributions
from Plots import plot_log_volumes_distributions
from Plots import plot_rsi

from Plots import plot_cppi_strategy
from Plots import plot_spread

from datetime import datetime
import pandas as pd
import ast

import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import chart_studio.plotly as py
import chart_studio
import numpy as np
import pandas as pd
chart_studio.tools.set_credentials_file(username='Crapotca', api_key='M4RwlPZfd7zrzjsl5aU8')

# Suppress warnings to make the output cleaner
import warnings
warnings.filterwarnings("ignore")

date_threshold = datetime(2017, 8, 21)
crypto_symbols = ['bitcoin', 'dogecoin', 'ethereum-classic', 'ethereum', 'litecoin', 'monero', 'ripple']
label = ['Bitcoin Analysis', 'Dogecoin Analysis', 'Ethereum Classic Analysis', 'Ethereum Analysis', 'Litecoin Analysis', 'Monero Analysis', 'Ripple Analysis']

model_data = read_model_data(crypto_symbols)
search_data = read_google_data(crypto_symbols)
cppi_data = read_cppi_data(crypto_symbols)

chmm_scores = pd.read_csv('hmm_scores.csv')
chmm_scores.set_index('Crypto', inplace=True)
chmm_best = pd.read_csv('hmm_best_models.csv')
chmm_best.set_index('Crypto', inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

returns_volatility = pd.DataFrame(columns=['Mean Returns (Weekly)', 'Realized Volatility (Weekly)',
                                           'Mean Returns (Monthly)', 'Realized Volatility (Monthly)',
                                           'Mean Returns (Yearly)', 'Realized Volatility (Yearly)'])

normality_results_data = pd.DataFrame(columns=['KS Test on Returns', 'Returns Skewness', 'Returns Kurtosis',
                                           'KS Test on Volumes', 'Volumes Skewness', 'Volumes Kurtosis'])


# Layout of the dashboard
app.layout = html.Div([

    dcc.Slider(
        id='crypto-slider-top',
        min=0,
        max=len(crypto_symbols) - 1,
        value=0,
        marks={i: crypto_symbol for i, crypto_symbol in enumerate(crypto_symbols)}),

    html.H1(id='dashboard-title', style={'text-align': 'center'}),

    dcc.Graph(id='crypto-plot'),

    dcc.Graph(id='google-plot'),

    html.Div([
        html.H2("Implied Means and Volatilities", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='returns-volatility-table',
            columns=[
                        {'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.3f'}}
                        for col in returns_volatility.columns
                    ],
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'}
        )]),

    html.Div(style={'height': '20px'}),

    dcc.Graph(id='density-plot'),

    html.Div(style={'height': '20px'}),

    html.Div([
            html.H2("Normal Distribution Test", style={'text-align': 'center'}),
            dash_table.DataTable(
                id='normality-results-table',
                columns=[
                    {'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.4f'}}
                    for col in normality_results_data.columns
                ],
                style_cell={'textAlign': 'center'},
                style_table={'margin-left': 'auto', 'margin-right': 'auto'}
            )]),

    dcc.Graph(id='qq-plot'),

    html.Div([
        html.H2("Hidden Markov Model State Scores", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='score-table',
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    html.Div(style={'height': '20px'}),

    dcc.Graph(id='state-plot'),

    html.Div(style={'height': '20px'}),

    html.Div([
        html.H2("Hidden Markov Model State Statistics", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='state-statistics',
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    html.Div(style={'height': '20px'}),

    dcc.Graph(id='heatmap'),

    dcc.Graph(id='return-plot'),

    html.Div(style={'height': '20px'}),

    html.Div([
        html.H2("Log Likelihood Analysis", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='ll',
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    dcc.Graph(id='volume-plot'),

    html.Div(style={'height': '20px'}),

    html.Div([
        html.H2("Augmented Dickey-Fuller Test", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='adf',
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    html.Div(style={'height': '40px'}),

    dcc.Graph(id='rsi-plot'),

    html.Div([
        html.H2("Applied Expected Values and Multipliers", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='cppi-table',
            columns=[{'name': col, 'id': col} for col in ['States', 'Expected Values', 'Multipliers']],
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    html.Div(style={'height': '20px'}),

    dcc.Graph(id='cppi-strategy-plot'),

    html.Div([
        html.H2("Investing Strategy Summary", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='summary-table',
            style_cell={'textAlign': 'center'},
            style_table={'margin-left': 'auto', 'margin-right': 'auto'})]),

    html.Div(style={'height': '20px'}),

    dcc.Graph(id='spread-plot'),

    html.Div(style={'height': '20px'}),

    dcc.Slider(
        id='crypto-slider-bottom',
        min=0,
        max=len(crypto_symbols) - 1,
        value=0,
        marks={i: crypto_symbol for i, crypto_symbol in enumerate(crypto_symbols)}),])


@app.callback(
    [Output('dashboard-title', 'children'),
        Output('crypto-plot', 'figure'),
        Output('google-plot', 'figure'),
        Output('returns-volatility-table', 'data'),
        Output('density-plot', 'figure'),
        Output('normality-results-table', 'data'),
        Output('qq-plot', 'figure'),
        Output('score-table', 'columns'),
        Output('score-table', 'data'),
        Output('state-plot', 'figure'),
        Output('state-statistics', 'data'),
        Output('heatmap', 'figure'),
        Output('return-plot', 'figure'),
        Output('ll', 'data'),
        Output('volume-plot', 'figure'),
        Output('adf', 'data'),
        Output('rsi-plot', 'figure'),
        Output('cppi-table', 'data'),
        Output('cppi-strategy-plot', 'figure'),
        Output('summary-table', 'data'),
        Output('spread-plot', 'figure')],
        Output('crypto-slider-top', 'value'),
        Output('crypto-slider-bottom', 'value'),
    [Input('crypto-slider-top', 'value'),
     Input('crypto-slider-bottom', 'value')])


def update_crypto_plots(top_slider_value, bottom_slider_value):
    ctx = dash.callback_context

    if not ctx.triggered:
        selected_crypto_index = 0
        top_slider_value = bottom_slider_value = selected_crypto_index
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'crypto-slider-top':
            selected_crypto_index = top_slider_value
            bottom_slider_value = top_slider_value
        else:
            selected_crypto_index = bottom_slider_value
            top_slider_value = bottom_slider_value

    selected_crypto_symbol = 'bitcoin'
    #selected_crypto_symbol = crypto_symbols[selected_crypto_index]
    selected_label = label[selected_crypto_index]

    selected_search_data = search_data[selected_crypto_symbol]
    selected_search_data.index = pd.to_datetime(selected_search_data.index)
    selected_search_data = selected_search_data[selected_crypto_symbol]

    selected_model_data = model_data[selected_crypto_symbol]
    selected_model_data.index = pd.to_datetime(selected_model_data.index)

    selected_price_data = selected_model_data['price']

    selected_log_data = pd.DataFrame({
        'log_return': selected_model_data['log_return'],
        'log_volume_variation': selected_model_data['log_volume_variation'],
        'states': selected_model_data['Decoded_States']
    })
    unique_states = sorted(selected_model_data['Decoded_States'].unique())

    selected_scores = chmm_scores.loc[selected_crypto_symbol]
    selected_scores_columns = []
    for i, col in enumerate(selected_scores.columns):
        if i < 3:
            selected_scores_columns.append({'name': col, 'id': col, 'type': 'numeric'})
        else:
            selected_scores_columns.append({'name': col, 'id': col, 'type': 'numeric', 'format': {'specifier': '.4f'}})

    selected_chmm = chmm_best.loc[selected_crypto_symbol]

    transmat_str = selected_chmm['Transmat']
    transmat = ast.literal_eval(transmat_str)
    startprob = selected_chmm['Startprob']
    startprob = ast.literal_eval(startprob)
    means_str = selected_chmm['Means']
    means = np.array([mean[0] for mean in ast.literal_eval(means_str)])

    cppi_table = create_state_multiplier_table(transmat, means)

    selected_cppi = cppi_data[selected_crypto_symbol]

    title = f'{selected_label}'

    crypto_plot = plot_price_chart(selected_price_data)
    google_plot = plot_searches_chart(selected_search_data)
    density_fig = plot_density_figure(selected_log_data)
    qq_fig = plot_qq_figure(selected_log_data)
    returns_volatility = calculate_returns_volatility(selected_price_data)
    normality_results_data = test_normality(selected_log_data)

    state_plot = states_plot(selected_model_data, unique_states)
    heatmap = plot_heatmap(transmat, startprob)
    state_statistics_df = create_state_statistics(selected_model_data)
    return_plot = plot_log_returns_distributions(selected_log_data, selected_chmm)
    volume_plot = plot_log_volumes_distributions(selected_log_data, selected_chmm)
    rsi_plot = plot_rsi(selected_search_data, selected_price_data)

    adf = calculate_adf(selected_price_data)
    ll = calculate_log_likelihoods(selected_model_data, selected_chmm)

    cppi_strategy_plot = plot_cppi_strategy(selected_cppi)
    spread_plot = plot_spread(selected_cppi)
    summary_data = create_summary_table(selected_cppi)


    py.plot(crypto_plot, filename=f'{selected_crypto_symbol}graph1', auto_open=True)
    py.plot(state_plot, filename=f'{selected_crypto_symbol}graph2', auto_open=True)
    py.plot(heatmap, filename=f'{selected_crypto_symbol}graph3', auto_open=True)
    py.plot(rsi_plot, filename=f'{selected_crypto_symbol}graph4', auto_open=True)
    py.plot(cppi_strategy_plot, filename=f'{selected_crypto_symbol}graph5', auto_open=True)
    py.plot(spread_plot, filename=f'{selected_crypto_symbol}graph6', auto_open=True)

    return (title,
            crypto_plot,
            google_plot,
            returns_volatility.to_dict('records'),
            density_fig,
            normality_results_data.to_dict('records'),
            qq_fig,
            selected_scores_columns,
            selected_scores.to_dict('records'),
            state_plot,
            state_statistics_df.to_dict('records'),
            heatmap,
            return_plot,
            ll.to_dict('records'),
            volume_plot,
            adf.to_dict('records'),
            rsi_plot,
            cppi_table.to_dict('records'),
            cppi_strategy_plot,
            summary_data.to_dict('records'),
            spread_plot,
            top_slider_value,
            bottom_slider_value)



# Run the Dash app (if necessary)
if __name__ == '__main__':
    app.run_server(debug=True)
