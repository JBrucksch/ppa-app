import dash
from dash import dcc, html, register_page
import dash_bootstrap_components as dbc

def output_info():
    return html.Div([
        html.H1("Results", className='header-title-2'),
    html.Hr(style={"borderColor": "blue", "borderWidth": "2px"}),

    html.H3("Battery Usage", className='header-style-2'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Energy Throughput", className="card-title"),
                    html.H6([html.Span(id="battery-energy"), " MWh"], className="card-subtitle"),
                    html.P("Total energy processed by the battery system", className="card-text")
                ]),
                dbc.Tooltip("This is the total amount of energy that has passed through the battery.", target="battery-energy"),
            ], className="m-3 shadow-sm"),  # Added margin and shadow
            dbc.Card([
                dbc.CardBody([
                    html.H5("Equivalent Full Cycles", className="card-title"),
                    html.H6([html.Span(id="battery-efc"), " cycles/year"], className="card-subtitle"),
                    html.P("Number of full charge-discharge cycles per year.", className="card-text")
                ]),
                dbc.Tooltip("This represents how many times the battery has been fully charged and discharged.", target="battery-efc"),
            ], className="m-3 shadow-sm"),  # Added shadow
            dbc.Card([
                dbc.CardBody([
                    html.H5("Average SOC", className="card-title"),
                    html.H6([html.Span(id="battery-SOC"), " %"], className="card-subtitle"),
                    html.P("Average State of Charge of the battery.", className="card-text")
                ]),
                dbc.Tooltip("Average percentage charge that the battery holds.", target="battery-SOC"),
            ], className="m-3 shadow-sm")
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Savings", className="card-title"),
                    html.H6([html.Span(id="battery-savings"), " €"], className="card-subtitle"),
                    html.P("Cost savings from using battery storage.", className="card-text")
                ]),
                dbc.Tooltip("Savings generated through optimal battery usage.", target="battery-savings"),
            ], className="m-3 shadow-sm"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Amortization", className="card-title"),
                    html.H6([html.Span(id="battery-ammort"), " years"], className="card-subtitle"),
                    html.P("Amortization time of the battery system.", className="card-text")
                ]),
                dbc.Tooltip("Breakdown of battery cost over its useful life.", target="battery-ammort"),
            ], className="m-3 shadow-sm"),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Warranty Usage", className="card-title"),
                    html.H6([html.Span(id="battery-warranty"), " %"], className="css-subtitle"),
                    html.P("Based on a 10 year and 6000 EFC warranty.", className="card-text")
                ]),
                dbc.Tooltip("Tracks how much of the battery’s warranty has been utilized.", target="battery-warranty"),
            ], className="m-3 shadow-sm")
        ], width=4)
    ], justify="center"),

    html.H3("Financial Results", className='header-style-2'),
    dbc.Row([dbc.Col([dcc.Graph(id="financial-plot-1")], width=12)], justify="evenly"),

    html.H3("Market Autarky", className='header-style-2'),
    dbc.Row([
        dbc.Col([dcc.Graph(id="donut-plot-1-1")], width=4),
        dbc.Col([dcc.Graph(id="donut-plot-1-2")], width=4),
        dbc.Col([dcc.Graph(id="donut-plot-1-3")], width=4),
    ], justify="evenly"),


    html.H3("PPA Usage", className='header-style-2'),
    dbc.Row([
        dbc.Col([dcc.Graph(id="donut-plot-3-1")], width=4),
        dbc.Col([dcc.Graph(id="donut-plot-3-2")], width=4),
    ], justify="evenly"),
 # Add margin-top for spacing from previous elements if needed

], className='container', id='output-content', style={"display": "none", "padding": "20px", "backgroundColor": "#f8f9fa"})
