import ast
from scipy.stats import norm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from statsmodels.graphics.gofplots import qqplot

##### Raw Data Plots #####

def plot_price_chart(selected_price_data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=selected_price_data.index,
            y=selected_price_data,
            mode='lines',
            name='Price',
            line=dict(color='black')
        )
    )
    fig.update_layout(
        title={
            'text': '<b>Price Chart</b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5
        },
        xaxis_title="<b>Date<b>",
        yaxis_title="<b>Price<b>",
        showlegend=False,
        height=550
    )
    return fig

def plot_searches_chart(selected_search_data):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=selected_search_data.index,
            y=selected_search_data,
            mode='lines',
            name='Price',
            line=dict(color='black')
        )
    )
    fig.update_layout(
        title={
            'text': '<b>Google Searches Chart</b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5
        },
        xaxis_title="<b>Date<b>",
        yaxis_title="<b>Searches<b>",
        showlegend=False
    )
    return fig

def plot_density_figure(selected_log_data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        '<b>Density Plot of Log Returns</b>',
        '<b>Density Plot of Log Volumes</b>'
    ])

    def add_density_plot(data, col, label, bin_size):
        distplot_fig = ff.create_distplot([data], [label], show_rug=False, bin_size=bin_size, colors=['black'])
        histogram_trace = distplot_fig['data'][0]
        histogram_trace.marker.color = 'blue'
        histogram_trace.name = 'Frequency Histogram'
        kde_trace = distplot_fig['data'][-1]
        kde_trace.line.color = 'black'
        kde_trace.line.width = 2.5

        for trace in distplot_fig['data']:
            trace.showlegend = False
            fig.add_trace(trace, row=1, col=col)

        fig.add_trace(
            go.Scatter(
                x=np.linspace(data.min(), data.max(), 1000),
                y=norm.pdf(np.linspace(data.min(), data.max(), 1000), loc=data.mean(), scale=data.std()),
                mode='lines',
                line=dict(color='red', width=2.5),
                showlegend=False,
                name='Gaussian Distribution'
            ), row=1, col=col)

    add_density_plot(selected_log_data['log_return'], 1, 'Kernel Density Estimation - Returns', 0.015)
    add_density_plot(selected_log_data['log_volume_variation'], 2, 'Kernel Density Estimation - Volumes', 0.09)

    for annotation in fig.layout.annotations:
        annotation.update(font=dict(family='Times New Roman', size=22, color='black'), y=1.1)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', size=14, symbol='square'),
                             name='Frequency Histogram'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='black', width=2.5),
                             name='Kernel Density Distribution'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='red', width=2.5),
                             name='Estimated Gaussian Distribution'))

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.25,
            xanchor='center',
            yanchor='bottom',
            traceorder='normal',
            font=dict(family='Times New Roman', size=16, color='black')))

    return fig

def plot_qq_figure(selected_log_data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        '<b>Q-Q Plot of Log Returns<b>',
        '<b>Q-Q Plot of Log Volumes<b>'
    ])

    log_return = selected_log_data['log_return']
    log_volume_variation = selected_log_data['log_volume_variation']

    qqplot_data_returns = qqplot(log_return, line='s').gca().lines
    for i, line in enumerate(qqplot_data_returns):
        trace_name = 'Quantile Cumulative Distribution' if i == 0 else 'Gaussian Distribution'
        fig.add_trace({
            'type': 'scatter',
            'x': line.get_xdata(),
            'y': line.get_ydata(),
            'mode': 'markers' if i == 0 else 'lines',
            'marker': {'color': 'blue'},
            'line': {'color': 'red', 'width': 2.5},
            'name': trace_name
        }, row=1, col=1)

    fig.update_xaxes(title_text='<b>Theoretical Quantiles<b>', row=1, col=1)
    fig.update_yaxes(title_text='<b>Sample Quantiles<b>', row=1, col=1)

    qqplot_data_volumes = qqplot(log_volume_variation, line='s').gca().lines
    for i, line in enumerate(qqplot_data_volumes):
        trace_name = 'Quantile Cumulative Distribution' if i == 0 else 'Gaussian Distribution'
        fig.add_trace({
            'type': 'scatter',
            'x': line.get_xdata(),
            'y': line.get_ydata(),
            'mode': 'markers' if i == 0 else 'lines',
            'marker': {'color': 'blue'},
            'line': {'color': 'red', 'width': 2.5},
            'name': trace_name
        }, row=1, col=2)

    fig.layout.annotations[0].update(font=dict(family='Times New Roman', size=22, color='black'), y=1.1)
    fig.layout.annotations[1].update(font=dict(family='Times New Roman', size=22, color='black'), y=1.1)

    fig.update_xaxes(title_text='<b>Theoretical Quantiles<b>', row=1, col=2)
    fig.update_yaxes(title_text='<b>Sample Quantiles<b>', row=1, col=2)

    fig.update_layout(height=450, showlegend=False)

    return fig


##### Hidden Markov Plots #####

def states_plot(selected_model_data, unique_states):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.05)

    for state in unique_states:
        state_data = selected_model_data[selected_model_data['Decoded_States'] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data.index,
                y=state_data['price'],
                mode='markers',
                marker=dict(size=7, color=state_data['Decoded_States'], colorscale='Rainbow', cmin=0, cmax=len(unique_states)-1),
                name=f'Hidden State {state+1}',
                showlegend=True
            ), row=1, col=1
        )

    y_positions = np.linspace(0, 1, len(unique_states))
    for i, state in enumerate(unique_states):
        state_data = selected_model_data[selected_model_data['Decoded_States'] == state]
        fig.add_trace(
            go.Scatter(
                x=state_data.index,
                y=[y_positions[i]] * len(state_data),  # Posizione y specifica per ciascuno stato
                mode='markers',
                marker=dict(size=4, color=state_data['Decoded_States'], colorscale='Rainbow', cmin=0, cmax=len(unique_states)-1),
                showlegend=False
            ), row=2, col=1
        )

    fig.update_layout(
        title={
            'text': '<b>Hidden Markov Model Price Chart<b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        yaxis_title='<b>Price<b>',
        yaxis2=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.25,
            xanchor='center',
            yanchor='bottom',
            traceorder='normal',
            font=dict(family='Times New Roman', size=18, color='black'))
    )

    fig.update_xaxes(title_text='<b>Date<b>', showticklabels=True, row=2, col=1)
    fig.update_yaxes(title_text="<b>States<b>", row=2, col=1)

    return fig

def plot_heatmap(transmat,startprob):

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.07
    )

    heatmap = go.Heatmap(
        z=transmat,
        x=[f'State {i + 1}' for i in range(len(transmat))],
        y=[f'State {i + 1} ' for i in range(len(transmat))],
        coloraxis="coloraxis"
    )
    fig.add_trace(heatmap, row=1, col=1)

    for i, row in enumerate(transmat):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=j, y=i, text=f'{val:.2f}', showarrow=False,
                xref='x1', yref='y1', font=dict(color='white', size=12)
            )

    startprob_bar = go.Bar(
        x=[f'State {i + 1}' for i in range(len(startprob))],
        y=startprob, marker=dict(color='blue')
    )
    fig.add_trace(startprob_bar, row=2, col=1)

    fig.update_layout(
        title={
            'text': '<b>Transition Matrix Heatmap<b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        showlegend=False,
        height=550,
        margin=dict(l=100, r=100)
    )

    fig.update_yaxes(title_text='StartProb', row=2, col=1)

    return fig

def plot_log_returns_distributions(selected_log_data, selected_chmm):
    means = ast.literal_eval(selected_chmm['Means'])
    covariances = ast.literal_eval(selected_chmm['Covariances'])
    num_states = len(means)

    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Hidden State {i+1}" for i in range(num_states)], vertical_spacing=0.15)

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(color='black'))

    for i in range(num_states):
        state_data = selected_log_data[selected_log_data['states'] == i]
        log_return = state_data['log_return']

        distplot_fig = ff.create_distplot([log_return], [''], show_rug=False, bin_size=0.01, colors=['blue'])
        histogram_trace = distplot_fig['data'][0]
        histogram_trace.showlegend = False
        histogram_trace.name = 'Frequency Histogram'
        fig.add_trace(histogram_trace, row=1 if i < 3 else 2, col=1 + i % 3)

        kde_trace = distplot_fig['data'][-1]
        kde_trace.line = dict(color='black', width=2.5)
        kde_trace.showlegend = False
        kde_trace.name = 'Kernel Density Distribution'
        fig.add_trace(kde_trace, row=1 if i < 3 else 2, col=1 + i % 3)

        mean_log_return, cov_log_return = means[i][0], covariances[i][0][0]
        x_values_return = np.linspace(min(log_return), max(log_return), 200)
        y_values_return = norm.pdf(x_values_return, mean_log_return, np.sqrt(cov_log_return))
        gaussian_trace = go.Scatter(x=x_values_return, y=y_values_return, mode='lines', line=dict(color='red', width=2.5))
        gaussian_trace.showlegend = False
        gaussian_trace.name = 'Estimated Gaussian Distribution'
        fig.add_trace(gaussian_trace, row=1 if i < 3 else 2, col=1 + i % 3)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', size=14, symbol='square'),
                             name='Frequency Histogram'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='black', width=2.5),
                             name='Kernel Density Distribution'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='red', width=2.5),
                             name='Estimated Gaussian Distribution'))

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.25,
            xanchor='center',
            yanchor='bottom',
            font=dict(family='Times New Roman', size=17, color='black')
        ),
        title={
            'text': '<b>Hidden Markov Model Returns Distributions</b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        margin=dict(t=100)
    )

    return fig

def plot_log_volumes_distributions(selected_log_data, selected_chmm):
    means = ast.literal_eval(selected_chmm['Means'])
    covariances = ast.literal_eval(selected_chmm['Covariances'])
    num_states = len(means)

    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"Hidden State {i+1}" for i in range(num_states)], vertical_spacing=0.15)

    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].update(font=dict(color='black'))

    for i in range(num_states):
        state_data = selected_log_data[selected_log_data['states'] == i]
        log_volume = state_data['log_volume_variation']

        distplot_fig = ff.create_distplot([log_volume], [''], show_rug=False, bin_size=0.075, colors=['blue'])
        histogram_trace = distplot_fig['data'][0]
        histogram_trace.showlegend = False
        histogram_trace.name = 'Frequency Histogram'
        fig.add_trace(histogram_trace, row=1 if i < 3 else 2, col=1 + i % 3)

        kde_trace = distplot_fig['data'][-1]
        kde_trace.line = dict(color='black', width=2.5)
        kde_trace.showlegend = False
        kde_trace.name = 'Kernel Density Distribution'
        fig.add_trace(kde_trace, row=1 if i < 3 else 2, col=1 + i % 3)

        mean_log_volume, cov_log_volume = means[i][1], covariances[i][1][1]
        x_values_volume = np.linspace(min(log_volume), max(log_volume), 200)
        y_values_volume = norm.pdf(x_values_volume, mean_log_volume, np.sqrt(cov_log_volume))
        gaussian_trace = go.Scatter(x=x_values_volume, y=y_values_volume, mode='lines', line=dict(color='red', width=2.5))
        gaussian_trace.showlegend = False
        gaussian_trace.name = 'Estimated Gaussian Distribution'
        fig.add_trace(gaussian_trace, row=1 if i < 3 else 2, col=1 + i % 3)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                             marker=dict(color='blue', size=14, symbol='square'),
                             name='Frequency Histogram'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='black', width=2.5),
                             name='Kernel Density Distribution'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color='red', width=2.5),
                             name='Estimated Gaussian Distribution'))

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.25,
            xanchor='center',
            yanchor='bottom',
            font=dict(family='Times New Roman', size=17, color='black')
        ),
        title={
            'text': '<b>Hidden Markov Model Volumes Distributions</b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        margin=dict(t=150)
    )

    return fig


def plot_rsi(selected_search_data, selected_price_data):

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04)

    fig.add_trace(
        go.Scatter(
            x=selected_price_data.index,
            y=selected_price_data,
            mode='lines',
            name='Price',
            line=dict(color='black')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=selected_search_data.index,
            y=selected_search_data,
            mode='lines',
            name='Searches',
            line=dict(color='black')
        ),
        row=2, col=1
    )

    for index, value in selected_search_data.items():
        if 20 <= value <= 40:
            color = "cyan"
        elif 40 < value <= 60:
            color = "blue"
        elif value > 60:
            color = "red"
        else:
            continue

        fig.add_vrect(
            x0=index, x1=index + pd.Timedelta(days=7),
            fillcolor=color, opacity=0.6,
            layer="below", line_width=0,
            row=1, col=1
        )

    fig.add_hline(y=75, line=dict(color='blue', width=1, dash='dot'), row=2, col=1)
    fig.add_hline(y=25, line=dict(color='blue', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(
        title={
            'text': '<b>Herding Chart with Google RSI<b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        yaxis_title='<b>Price<b>',
        yaxis2=dict(
            showgrid=False,
            zeroline=False
        ),
        showlegend=False,
        height=600,
    )
    fig.update_xaxes(title_text='<b>Date<b>', showticklabels=True, row=2, col=1)
    fig.update_yaxes(title_text="<b>RSI<b>", row=2, col=1)


    return fig


def plot_cppi_strategy(selected_cppi):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04)

    # Plot 'Strategy' in the main plot
    fig.add_trace(
        go.Scatter(
            x=selected_cppi.index,
            y=selected_cppi['Strategy'],
            mode='lines',
            name='CPPI Strategy',
            line=dict(width=2.5)
        ),
        row=1, col=1
    )

    # Plot 'Strategy' in the main plot
    fig.add_trace(
        go.Scatter(
            x=selected_cppi.index,
            y=selected_cppi['HOLD'],
            mode='lines',
            name='Holding Strategy',
            line=dict(width=2.5)
        ),
        row=1, col=1
    )

    # Plot 'Backup' in the subplot
    fig.add_trace(
        go.Scatter(
            x=selected_cppi.index,
            y=selected_cppi['Backup'],
            mode='lines',
            name='Locked-in Profits',
            line=dict(color='black', width=2.5)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.25,
            xanchor='center',
            yanchor='bottom',
            traceorder='normal',
            font=dict(family='Times New Roman', size=18, color='black')),
        margin=dict(t=100)
    )

    fig.update_layout(
        title={
            'text': '<b>Proposed CPPI Strategy Chart<b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        yaxis_title='<b>Portfolio Value<b>',
        height=600,
    )

    fig.update_xaxes(title_text='<b>Date<b>', showticklabels=True, row=2, col=1)
    fig.update_yaxes(title_text="<b>Backup<b>", row=2, col=1)


    return fig

def plot_spread(selected_cppi):
    fig = go.Figure()

    # Plot 'SPREAD' for the selected cryptocurrency
    fig.add_trace(
        go.Scatter(
            x=selected_cppi.index,
            y=selected_cppi['SPREAD'],
            mode='lines',
            name='Spread',
            line=dict(width=2.5)
        )
    )

    # Plot for 'Zero Return'
    fig.add_trace(
        go.Scatter(
            x=selected_cppi.index,
            y=[0] * len(selected_cppi.index),
            mode='lines',
            name='Zero Line',
            showlegend=False,
            line=dict(color='black', width=2.5, dash='dash')
        )
    )

    fig.update_layout(
        title={
            'text': '<b>Spread between Strategy and Holding<b>',
            'font': dict(family='Times New Roman', size=26, color='black'),
            'x': 0.5,
        },
        yaxis_title='<b>Spread<b>',
        showlegend=False,
        height=600,
    )
    fig.update_xaxes(title_text='<b>Date<b>', showticklabels=True)


    return fig
