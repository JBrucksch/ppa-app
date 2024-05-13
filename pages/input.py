import dash_bootstrap_components as dbc
import dash
from dash import dcc, html, register_page
import dash_leaflet as dl
import dash_daq as daq


def input_info():
    my_icon = dict(
        iconUrl='./assets/click_icon.png',
        iconSize=[45, 45],
        iconAnchor=[22.5, 36],
    )
    return html.Div([
        dbc.Row([
            dbc.Col(
                [
                    # dbc.Row([
                    #     html.Div("PPA Map", style={"fontWeight": "bold"})
                    # ]),
                    dl.Map(id="PPA-map",
                           children=[dl.TileLayer(),
                                     dl.LayerGroup(id="marker-layer", children=[]),
                                     dl.Marker(id="map-click-point", icon=my_icon, position=(0, 0))],
                           center=[50, 10], zoom=6, style={"height": "100%"},

                           ),
                    dcc.Store(id="marker-store", data=[]),
                ], className='container card_container'
            ),

            dbc.Col([
                html.Div([
                    #html.H5("PPA Config"),
                    html.H5("Define PPA Contract", className='card-title'),
                    # Add your line here
                    html.Hr(style={"borderColor": "blue", "borderWidth": "2px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["PPA Price", html.I(className="bi bi-info-circle me-2 icon",
                                                           id="icon_ppa_price")]),
                            dbc.Input(id="ppa_price", type="number", value=0.12)]),
                        dbc.Tooltip(
                            "Price for electricity generated by PPA system.",
                            target="icon_ppa_price",
                            placement='right',
                            className='info_tooltip'
                        ),

                        dbc.Col([
                            dbc.Label(["PPA Size [MW]", html.I(className="bi bi-info-circle me-2 icon",
                                                               id="icon_ppa_size")]),
                            dbc.Input(id="ppa_size", type="number", value=1)]),
                        dbc.Tooltip(
                            "Maximum electricity capacity.",
                            target="icon_ppa_size",
                            placement='right',
                            className='info_tooltip'
                        ),
                    ], className='param-line'),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Choose technology", html.I(className="bi bi-info-circle me-2 icon",
                                                                   id="icon_tech-dropdown")]),
                            dcc.Dropdown(["PV", "Wind"], "PV", id="tech-dropdown")]),
                        dbc.Tooltip(
                            "Select energy source.",
                            target="icon_tech-dropdown",
                            placement='right',
                            className='info_tooltip'
                        ),
                    ], className='param-line'),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Latitude", html.I(className="bi bi-info-circle me-2 icon",
                                                          id="icon_lat")]),
                            dbc.Input(id="lat", type="number", value=0)]),
                        dbc.Tooltip(
                            "North-south position.",
                            target="icon_lat",
                            placement='right',
                            className='info_tooltip'
                        ),
                        dbc.Col([
                            dbc.Label(["Longitude", html.I(className="bi bi-info-circle me-2 icon",
                                                           id="icon_lon")]),
                            dbc.Input(id="lon", type="number", value=0)]),
                        dbc.Tooltip(
                            "East-west position.",
                            target="icon_lon",
                            placement='right',
                            className='info_tooltip'
                        ),
                    ], className='param-line'
                    ),
                    dbc.Row([
                        html.Div(id="added_power_profiles_container")
                    ], className='param-line'
                    ),
                    dbc.Row([
                        # Add the loading spinner
                        dbc.Spinner(html.Div(id="loading-profile"), color="primary", type="grow"),
                        dbc.Col([dbc.Button("Get Power Profile", id="download-button", className="halfwidth")])
                    ], className='param-line cal-row'
                    ),
                ], className='container card_container'),

                # html.Hr(style={'borderWidth': "0.2vh", "width": "100%", "color": "#000000"}),

                html.Div(id="output-data-download"),

                # Use dcc.Store to store the data created in the callback
                dcc.Store(id="power-profile-store"),

                # Use dcc.Store to store the data created in the callback
                dcc.Store(id="pv-profile-store"),

                html.Div(style={"marginBottom": "1em"}),

                html.Div([
                    html.H5("Electricity Demand", className='card-title-demand'),
                    html.Div([
                        # Add your line here
                        html.Hr(style={"borderColor": "blue", "borderWidth": "2px"}),
                        # html.Div("You don't have a yearly profile."),
                        daq.BooleanSwitch(id='electricity_demand_boolean_switch', on=True,
                                          label="I don't have a yearly profile."),

                        html.Div(style={"marginBottom": "0.5em"}),
                        # during testing, change the max_size, it should be 5MB
                        dcc.Upload(
                            html.Div([
                                dbc.Button([
                                    "Upload Load File",
                                    html.I(className="bi bi-info-circle me-2 icon", id="icon_upload_button")
                                ], id="upload-button", className="halfwidth"),
                            ]),
                            id="upload-load-profile-data",
                            multiple=False, max_size=6 * 1024 * 1024,
                            className='cal-row'
                        ),
                        dbc.Tooltip(
                            "Please upload a file in 5 MB, including date and value.",
                            target="icon_upload_button",
                            placement='right',
                            className='info_tooltip'
                        ),

                        html.Div([
                            dbc.Label(["Yearly Electricity Demand [MWh]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_yearly_demand")]),
                            dbc.Input(id="yearly_demand", type="number", value=8760),
                            dbc.Tooltip(
                                "Duration of investment.",
                                target="icon_yearly_demand",
                                placement='right',
                                className='info_tooltip'
                            ),

                            html.Div(style={"marginBottom": "0.5em"}),
                            html.Div([
                                dbc.Button("Generate Load File", id="generate_load_file_button",
                                           color="primary"),
                            ], className='cal-row'),

                            dcc.Store(id="generate_load_file_store", data=[]),
                        ], id='yearly_demand_box', style={"display": "none"}
                        ),
                    ], id='load_profile_box'
                    ),
                    html.Div(id="upload-load-profile-message", className='cal-row'),
                    html.Div(id="upload-load-profile-average", className='cal-row'),
                    html.Div(id="upload-load-profile-mean", className='cal-row'),
                ], className='container card_container'),

                html.Div([
                    html.H5("Battery Design", className='card-title-demand'),
                    dbc.Row([
                        # Add your line here
                        html.Hr(style={"borderColor": "blue", "borderWidth": "2px"}),
                        dbc.Col([
                            dbc.Label(["Battery size [kWh]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_battery_size")]),
                            dbc.Input(id="battery_size", type="number", value=1000),
                            dbc.Tooltip(
                                "The storage capacity of the battery system.",
                                target="icon_battery_size",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),

                        dbc.Col([
                            dbc.Label(["Battery costs [Euro/kWh]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_battery_cost")]),
                            dbc.Input(id="battery_cost", type="number", value=500),
                            dbc.Tooltip(
                                "The price per unit of battery storage capacity.",
                                target="icon_battery_cost",
                                placement='right',
                                className='info_tooltip'
                            ),

                        ]),
                    ], className='param-line'),

                    dbc.Row([
                        # Add your line here
                        dbc.Col([
                            dbc.Label(["C-Rate [1/h]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_crate")]),
                            dbc.Input(id="bat_crate", type="number", value=1),
                            dbc.Tooltip(
                                "C-Rate of storage system.",
                                target="icon_crate",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),

                        dbc.Col([
                            dbc.Label(["Efficiency [%]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_eff")]),
                            dbc.Input(id="bat_eff", type="number", value=95),
                            dbc.Tooltip(
                                "Charging and discharging efficiency of battery system.",
                                target="icon_eff",
                                placement='right',
                                className='info_tooltip'
                            ),

                        ]),
                    ], className='param-line'),

                    daq.BooleanSwitch(id='peak_shaving_boolean_switch', on=False,
                                          label="Include Peak Shaving"),

                    html.Div([
                        dbc.Row([
                        
                        dbc.Col([
                            dbc.Label(["Allowed Peak Power [kW]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_peak_power")]),
                            dbc.Input(id="max_peak_power", type="number", value=1000),
                            dbc.Tooltip(
                                "The battery will try to shave power demand above that value.",
                                target="icon_eff",
                                placement='right',
                                className='info_tooltip'
                            ),

                        ]),
                    ], className='param-line')]
                    ,id='peak_shaving_box', style={"display": "block"}
                    ),

                ], className='container card_container'),

                html.Div(style={"marginBottom": "1em"}),
                html.Div([
                    html.H5("General Configuration", className='card-title-general'),
                    dbc.Row([
                        # Add your line here
                        html.Hr(style={"borderColor": "blue", "borderWidth": "2px"}),
                        dbc.Col([
                            dbc.Label(["Supplier price [Euro/kWh]", html.I(className="bi bi-info-circle me-2 icon", id="icon_grid_price")]),
                            dbc.Input(id="grid_price", type="number", value=0.3),
                            dbc.Tooltip(
                                "Price for electricity in public markt.",
                                target="icon_grid_price",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),

                        dbc.Col([
                            dbc.Label(["Feed-In Tarif [Euro/kWh]",
                                   html.I(className="bi bi-info-circle me-2 icon", id="icon_injection_price")]),
                            dbc.Input(id="injection_price", type="number", value=0.05),
                            dbc.Tooltip(
                                "Price for selling excess electricity.",
                                target="icon_injection_price",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),
                    ], className='param-line'),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Peak Power Costs [Euro/kW]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_peak_demand_price")]),
                            dbc.Input(id="peak_demand_price", type="number", value=120),
                            dbc.Tooltip(
                                "The cost of electricity during peak usage times.",
                                target="icon_peak_demand_price",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),

                        dbc.Col([
                            dbc.Label(["Interest rate [%]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_interest_rate")]),
                            dbc.Input(id="interest_rate", type="number", value=8),
                            dbc.Tooltip(
                                "Annual interest rate applied to financing.",
                                target="icon_interest_rate",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),
                    ], className='param-line'),


                    dbc.Row([
                        dbc.Col([
                            dbc.Label(["Investment horizon [a]",
                                       html.I(className="bi bi-info-circle me-2 icon", id="icon_horizon")]),
                            dbc.Input(id="horizon", type="number", value=20),
                            dbc.Tooltip(
                                "Duration of investment.",
                                target="icon_horizon",
                                placement='right',
                                className='info_tooltip'
                            ),
                        ]),
                    ], className='param-line'),
                ], className='container card_container'),

                html.Div(style={"marginBottom": "1em"}),

                dbc.Modal([
                    dbc.ModalHeader("PPA Contract Information"),
                    dbc.ModalBody([
                        dbc.Label("Generator Type"),
                        dcc.Dropdown(
                            id="generator-type-dropdown",
                            options=[
                                {"label": "Wind Generator", "value": "Wind"},
                                {"label": "PV Generator", "value": "PV"}
                            ],
                            value="PV"
                        ),

                        dbc.Label("PPA Price"),
                        dbc.Input(id="ppa-modal-price", type="number"),

                        dbc.Label("Generator Size (MW)"),
                        dbc.Input(id="generator-size", type="number"),

                        dbc.Label("Latitude"),
                        dbc.Input(id="ppa-modal-lat", type="number"),

                        dbc.Label("Longitude"),
                        dbc.Input(id="ppa-modal-lon", type="number"),
                    ]),
                    dbc.ModalFooter([
                        dbc.Button("Get Profile", id="get-profile-button", color="primary"),
                        dbc.Button("Save Information", id="save-info-button", color="success"),
                        dbc.Button("Close", id="close-ppa-modal", color="secondary"),
                    ]),
                ], id="ppa-modal", centered=True)

            ], width=4)
        ]),

        dbc.Row([
            dbc.Spinner(html.Div(id="loading-output"), color="primary", type="grow"),
            dbc.Col([
                dbc.Button("Calculate", id="calculate-button", color="primary", size="lg")
            ], className='cal-row'
            ),

            dbc.Modal([
                dbc.ModalHeader("Message"),
                dbc.ModalBody([
                    html.Div(id="popup-text"),
                ]),
                # dbc.ModalFooter(
                #     dbc.Button("OK", id="close-popup", className="ml-auto")
                # ),
            ],
                id="popup",
                centered=True,
                is_open=False,
            )

        ]),

        html.Div(style={"marginBottom": "1em"}),

        html.Div([
            html.Div(
                "This data has been made available for commercial and non-commercial purposes under "
                "CC4.0-BY. Cite this website as source in any derived works.", id="cite", className='cal-row'),
            html.Div(id="text-field", style={"display": "none"}),
            dash.dash_table.DataTable(id="explanation-table",
                                      style_cell={"text_align": "left",
                                                  "color": "black",
                                                  "border": "1px solid rgb(0, 84, 159)"
                                                  },
                                      style_data_conditional=[{"if": {"state": "active"},
                                                               "backgroundColor": "white",
                                                               "border": "1px solid rgb(0, 84, 159)"}]),
        ],
            className="box"
        ),
    ], id='input-content')
