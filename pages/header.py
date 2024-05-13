import dash
from dash import html
import dash_bootstrap_components as dbc


def header_info():
    return [
            dbc.Col(html.Img(src=dash.get_asset_url('rwth_isea_en_rgb.png'), style={'width': '60%'}), width=4),
            dbc.Col(html.Img(src=dash.get_asset_url('Tool Logo.svg'), style={'width': '60%', 'margin': 'auto', 'display': 'block'}), 
                    width=4, 
                    style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            dbc.Col(html.Img(src=dash.get_asset_url('rwth_eerc_cmyk.png'), style={'width': '60%'}), width=4, style={'display': 'flex', 'justify-content': 'flex-end'})
        ]