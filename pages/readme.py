from dash import html, dcc
import dash_bootstrap_components as dbc


def readme_info():
    return dbc.Row([
        html.H1('Readme'),
        html.H3('Input data'),
        html.Div([
            html.Li('Power profiles: '),
        ]),
        html.H3('Optimization'),
        html.Div([
            html.Li('MILP optimization'),
            html.Li(
                'objective is to minimize annual costs for procuring electricity (list the cost components we include i.e. ppa costs, grid costs, battery costs)'),
            html.Li('What happens to excess energy --> sold to excess feed-in price given by user'),
            html.Li(
                'What happens if we need more electricity than we get from ppa? --> we get if from the grid with the grid costs the user gives'),
            html.Li('Solver: with link to cbc solver'),
        ]),
        html.H3('Other'),
        html.Div([
            html.Li(
                'how are the investment costs annualized? --> annuity factor (give link to wikipedia) including the investment horizon and interest rate the user specifies'),
            html.Li(
                'The tool only can give estimations on how much the electricity procurement really costs. Many uncertainties exist due to weather, price and load uncertainty for future scenarios'),
        ]),
        ], align='center', id='readme-content', className='container card_container readme'
    )
