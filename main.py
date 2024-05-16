import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, MATCH
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from toolbox import *
from plotly.subplots import make_subplots
import dash_leaflet as dl
import dash._callback_context as ctx
from pages.header import header_info
from pages.readme import readme_info
from pages.input import input_info
from pages.output import output_info
import base64
import io
import copy
import csv

app = dash.Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])

app.title = 'PPA-APP'

####################################################################################################################
# Layout
####################################################################################################################

app.layout = html.Div([
    
    dbc.Row(
        header_info()
    ,className='header-box',align='center'
    ),

    dcc.Store(id='load-profile-data-store'),

    dbc.Row([
        input_info()
    ], className='main-box'
    ),

    dbc.Row([
        output_info()
    ], className='main-box', align='center'
    ),
    ])

server = app.server

#######################################################################################################################
# CALLBACKS
#######################################################################################################################
pp_data = {}

# @app.callback(
#     Output("alert-1", "is_open"),
#     [Input("alert-1", "n_clicks")],
#     [State("alert-1", "is_open")],
# )
# def toggle_alert(n, is_open):
#     if n:
#         return not is_open
#     return is_open

@app.callback(
    Output("map-click-point", "position"),
    Output("lat", "value"),
    Output("lon", "value"),
    Input("PPA-map", "clickData"),
    prevent_initial_call=True
)
def get_lat_lon(clickData):
    if clickData is not None:
        lat = clickData["latlng"]["lat"]
        lon = clickData["latlng"]["lng"]
        return {"lat": lat, "lon": lon}, str(lat), str(lon)
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("marker-layer", "children"),
    Output("loading-profile", "children"),
    Input("download-button", "n_clicks"),
    State("ppa_price", "value"),
    State("ppa_size", "value"),
    State("tech-dropdown", "value"),
    State("lat", "value"),
    State("lon", "value"),
    State("added_power_profiles_container", "children"),
    State("marker-layer", "children"),
    prevent_initial_call=True,
)
def download_data(n_clicks, price, size, tech, lat, lon, current_content, marker_layer):
    """
    Calculates the Power Profile according to the parameters and saves it in PP storage.
    :param n_clicks:
    :param price:
    :param size:
    :param tech:
    :param lat:
    :param lon:
    :param current_content:
    :param marker_layer:
    :return:
    """
    global pp_data
    if n_clicks == 1:
        pp_data = {}

    lat = str(lat)
    lon = str(lon)
    pp_message_to_be_displayed = "Type: " + tech + "; Price: " + str(price) + " €/MWh; Size: " + str(size) + " MW           "

    new_power_profile = html.Div([pp_message_to_be_displayed,
                                  html.Button("Remove", id={"type": "remove-profile-button", "ind": n_clicks},
                                              n_clicks=0)
                                  ]
                                 )

    '''
    if current_content is not None:
        updated_content = [*current_content, new_power_profile]
    else:
        updated_content = [new_power_profile]
    '''

    #new_data = get_irradiance(float(lat), float(lon), size, tech)
    new_data = get_irradiance(float(lat), float(lon), size, tech)
    # new_data = pd.read_csv("test_data_pv.csv", index_col=0)
    # new_data = np.array(new_data)
    # new_data = new_data.flatten()

    icon_url = './assets/{}_icon.png'.format(tech)
    my_icon = dict(
        iconUrl=icon_url,
        iconSize=[45, 45],
        iconAnchor=[22.5, 36],
        tooltipAnchor=[20, 0]
    )
    new_marker = dl.Marker(
        position=[float(lat), float(lon)],
        id={"type": "marker_list", "ind": n_clicks},
        icon=my_icon,
        riseOnHover='true',
        children=html.Div([dl.Tooltip(children=new_power_profile, permanent='True', interactive='True',
                                      bubblingMouseEvents='True')])
    )
    marker_layer.append(new_marker)
    # print(marker_layer)

    pp_data[n_clicks] = [price, size, tech, float(lat), float(lon), new_data]
    # print(pp_data.keys())
    # print(pp_data)
    return marker_layer, ""


@app.callback(
    Output({"type": "marker_list", "ind": MATCH}, "opacity"),
    Output({"type": "marker_list", "ind": MATCH}, "children"),
    Input({"type": "remove-profile-button", "ind": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def delete_power_profile(_):
    i = ctx.context_value.get()["inputs_list"][0]["id"]["ind"]  # gets the key of the element to be deleted
    global pp_data
    pp_data.pop(i)
    return 0, []


'''
@app.callback(
    Output("marker-layer", "children", allow_duplicate=True),
    Input({"type": "remove-profile-button", "ind": ALL}, "n_clicks"),
    State("marker-layer", "children"),
    prevent_initial_call=True
)
def delete_marker(n_clicks, marker_data):
    clicked_dicts = [{i + 1: click} for i, click in enumerate(n_clicks)]
    indices_with_1 = [str(index + 1) for index, d in enumerate(clicked_dicts) if 1 in d.values()]

    new_markers = []
    for marker in marker_data:
        if marker["props"]["id"] not in indices_with_1:
            new_markers.append(marker)

    return new_markers
'''


@app.callback(
    Output('upload-load-profile-data', 'style'),
    Output('yearly_demand_box', 'style'),
    Input('electricity_demand_boolean_switch', 'on')
)
def yearly_demand_choose(on):
    if on:
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}
    
@app.callback(
    Output('peak_shaving_box', 'style'),
    Input('peak_shaving_boolean_switch', 'on')
)
def peak_shaving_choose(on):
    if on:
        return {"display": "block"}
    else:
        return {"display": "none"}

@app.callback(
    Output("upload-load-profile-message", "children", allow_duplicate=True),
    Output("load-profile-data-store", "data"),
    Input("upload-load-profile-data", "contents"),
    Input("upload-load-profile-data", "filename"),
    prevent_initial_call=True
)
def update_output(contents, file_name):
    if contents is not None and len(contents) > 6 * 1024 * 1024:
        return html.Div("File size exceeds the limit (5MB). Please upload a smaller file.", style={"color": "red"}), None
    else:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        load_profile = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=',')
        if len(load_profile.columns) != 2:
            load_profile = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';')
            if len(load_profile.columns) != 2:
                return html.Div("File mismatch.", style={"color": "red"}), None
                    
        if len(load_profile) < 1000:
            return html.Div("File should have more enteries.", style={"color": "red"}), None
        else:
            return f"{file_name} uploaded successfully", load_profile.to_json()

@app.callback(
    Output("upload-load-profile-message", "children", allow_duplicate=True),
    Output("upload-load-profile-average", "children", allow_duplicate=True),
    Output("upload-load-profile-mean", "children", allow_duplicate=True),
    Output("generate_load_file_store", "data"),
    Input("generate_load_file_button", "n_clicks"),
    State("yearly_demand", "value"),
    prevent_initial_call=True
)
def generate_load_file(n_clicks, yearly_demand):
    #load csv file from input_data folder
    orig_hourly_demand = pd.read_csv('input_data\power_hourly_braeuer15.csv', header=None)

    # rescale so that the sum of the demand is the same as the total demand
    yearly_demand_timeseries = orig_hourly_demand * yearly_demand * 1000 / orig_hourly_demand.sum()



    df = pd.DataFrame()
    each_value = (yearly_demand * 1000)/8760
    print("!!! Time series test.")
    time_series = pd.period_range(start ='2023-01-01 00:00:00',
                                      end ='2023-12-31 23:00:00', freq ='h')
    
    # change index to time series
    yearly_demand_timeseries.index = time_series

    df['date'] = time_series
    df['values'] = each_value
    # print("!!! df: ", df.set_index('date'))

    # calculate demand stats
    avg_demand = yearly_demand_timeseries.mean()
    max_demand = yearly_demand_timeseries.max()

    output_text = "Your Load File is generated successfully."
    avg_text = f"Average demand: {int(avg_demand)} kW."
    peak_text = f"Peak demand: {int(max_demand)} kW."

    return output_text, avg_text, peak_text, yearly_demand_timeseries.to_json()


# Callback to handle the calculation and box plot update
@app.callback(
    Output("popup-text", "children"),
    Output("popup", "is_open"),
    # Output("", ""),
    Output("loading-output", "children"),
    Output("loading-output", "style"),
    Output("battery-energy", "children"),
    Output("battery-efc", "children"),
    Output("battery-SOC", "children"),
    Output("battery-savings", "children"),
    Output("battery-ammort", "children"),
    Output("battery-warranty", "children"),
    Output("financial-plot-1", "figure"),
    Output("donut-plot-1-1", "figure"),
    Output("donut-plot-1-2", "figure"),
    Output("donut-plot-1-3", "figure"),
    #Output("donut-plot-2-1", "figure"),
    #Output("donut-plot-2-2", "figure"),
    #Output("donut-plot-2-3", "figure"),
    Output("donut-plot-3-1", "figure"),
    Output("donut-plot-3-2", "figure"),
    Output('output-content', 'style'),
    Input("calculate-button", "n_clicks"),
    State("grid_price", "value"),
    State("injection_price", "value"),
    State("ppa_price", "value"),
    State("ppa_size", "value"),
    State("peak_demand_price", "value"),
    State("max_peak_power", "value"),
    State("battery_size", "value"),
    State("bat_eff", "value"),
    State("bat_crate", "value"),
    State("battery_cost", "value"),
    State("interest_rate", "value"),
    State("horizon", "value"),
    # State("power-profile-store", "data"),  todo remove it from the structure too
    State("load-profile-data-store", "data"),
    State('electricity_demand_boolean_switch', 'on'),
    State('peak_shaving_boolean_switch', 'on'),
    State("generate_load_file_store", "data"),
    State('output-content', 'style'),
    # Add other states for checkboxes and numerical inputs here
    prevent_initial_call=True
)
def collect_input_data(n_clicks,  # checkbox_1_value,
                       grid_price, injection_price, ppa_price, ppa_size, peak_demand_price, max_peak_power,
                       battery_size, bat_eff, bat_crate, battery_cost, interest_rate, horizon, load_profile,
                       demand_boolean, ps_boolean, load_profile_store, show_output):
    if n_clicks is None:
        return dash.no_update
    print('!!! n_clicks:', n_clicks)

    # in this block, the popup is being handled
    if demand_boolean is False and load_profile is None:
        return "Please upload a Load Profile file.", True, None, None, None, None, None, None, None, None, None, show_output
    elif demand_boolean is True and len(load_profile_store) == 0:
        return "Please fill in the Yearly Electricity Demand.", True, None, None, None, None, None, None, None, None, None, show_output
    global pp_data
    pv_count, wind_count = 0, 0
    for key, pp in pp_data.items():
        if pp[2] == "PV":
            pv_count += 1
        elif pp[2] == "Wind":
            wind_count += 1
    if pv_count > 5 and wind_count > 10:
        message = f"At most 10 PPAs are allowed per technology. \nPlease remove {pv_count - 10} PV PPAs and {wind_count - 10} Wind PPAs."
        return message, True, None, None, None, None, None, None, None, None, None, show_output
    elif pv_count > 5:
        message = f"At most 10 PPAs are allowed per technology. \nPlease remove {pv_count - 10} PV PPAs."
        return message, True, None, None, None, None, None, None, None, None, None, show_output
    elif wind_count > 5:
        message = f"At most 10 PPAs are allowed per technology. \nPlease remove {wind_count - 10} Wind PPAs."
        return message, True, None, None, None, None, None, None, None, None, None, show_output


    loading_output = dbc.Spinner(color="primary", size="md")
    loading_style = {"display": "block", "margin": "auto"}

    if demand_boolean:
        load_profile = pd.read_json(load_profile_store)
    else: 
        load_profile = pd.read_json(load_profile)
        load_profile.set_index(load_profile.columns[0], inplace=True)
        # content_type, content_string = load_profile.split(',')
        # decoded = base64.b64decode(content_string)
        # #load_profile = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=delimiter)
        # delimiter = ';' if ';' in load_profile.columns[0] else ','
        # load_profile[load_profile.columns[0]] = load_profile[load_profile.columns[0]].str.split(delimiter).str[0]
        # load_profile.set_index(load_profile.columns[0], inplace=True)
    # load_profile = pd.read_csv("sekurit_total_kW1_2019_2022.csv", index_col=0)
    load_profile = edit_series(load_profile)
    load_profile[load_profile > 30000] = 5000


    # GET IRRADIANCE DATA BASED ON CSV
    # t_horizon = range(len(load_profile))
    t_horizon = 365 * 24

    power_profiles = []
    for key, power_profile in pp_data.items():
        power_profiles.append([x * ppa_size * 1000 for x in power_profile[5]])
    agg_power_profile = [sum(x) for x in zip(*power_profiles)]
    # agg_power_profile = pd.Series(agg_power_profile) todo look if everything ran fine without this

    # only for testing
    # agg_power_profile = agg_power_profile[0:t_horizon]
    # load_profile = load_profile[0:t_horizon]
    # price_profile = price_profile[0:t_horizon]

    input_dict = {}
    input_dict["price_grid"] = grid_price
    input_dict["injection_price"] = injection_price
    input_dict["price_pap"] = ppa_price
    input_dict["price_peak_load"] = peak_demand_price
    input_dict["battery_capacity"] = battery_size
    input_dict["price_bat"] = battery_cost
    input_dict["interest_rate"] = interest_rate / 100
    input_dict["investment_horizon"] = horizon
    input_dict["max_peak_power"] = max_peak_power
    input_dict["efficiency_bat"] = bat_eff / 100
    input_dict["crate_bat"] = bat_crate

    # opt_result_dict = get_opt_result_dict()
    result_dict = get_optimization_result(input_dict, t_horizon, pp_data, load_profile, ps_boolean)
    

    energy_throughput = round(result_dict['Storage']['energy_to_battery'], 1)
    efc = round(result_dict['Storage']['energy_to_battery']*1000 / input_dict["battery_capacity"], 1)
    avg_soc = round(0, 1)
    savings = round(result_dict['PPA']['annuity'] - result_dict['Storage']['annuity'], 1)
    ammort = round(input_dict["battery_capacity"] * input_dict["price_bat"] / savings, 1)
    warranty = round(efc*10/6000 * 100, 1)

    fig_cost_anls = plot_cost_analy(result_dict, input_dict)
    #fig_load_duration = plot_load_dur_curve(model_storage)
    #fig_pp_gentn = plot_elec_generated(copy.deepcopy(pp_data), ppa_size)
    #fig_pv, fig_wind = plot_individual_prod(copy.deepcopy(pp_data), ppa_size)
    #fig4 = plot_fig4(t_horizon, result_dict, load_profile, price_profile, agg_power_profile)
    #fig_emissions = plot_emissions(result_dict)
    #fig_consmp_dmnd = plot_consmp_dmnd(result_dict, load_profile)

    donut_plots_import, donut_plots_autarky, donut_plots_ppa = plot_donut(result_dict, load_profile, agg_power_profile, pp_data)

    # Hide the loading spinner after the calculation is complete
    loading_output = None
    loading_style = {"display": "none"}
    show_output = {"display": "block"}

    print("---GRAPHS PLOTTED---")

    return None, False, loading_output, loading_style, \
        energy_throughput,efc, avg_soc, savings, ammort, warranty,\
            fig_cost_anls, \
                donut_plots_autarky[2], donut_plots_autarky[1], donut_plots_autarky[0], \
                        donut_plots_ppa[1], donut_plots_ppa[0],show_output

@app.callback(
    Output("ppa-modal", "is_open"),
    # Input("open-ppa-modal", "n_clicks"),
    Input("close-ppa-modal", "n_clicks"),
    State("ppa-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(open_clicks, close_clicks, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "open-ppa-modal":
        return not is_open
    elif button_id == "close-ppa-modal":
        return False
    return is_open


@app.callback(
    Output("pv-profile-store", "data"),
    Input("get-profile-button", "n_clicks"),
    State("ppa-modal-lat", "value"),
    State("ppa-modal-lon", "value"),
    prevent_initial_call=True
)

def get_power_profile(n_clicks, lat, lon):
    # Call your external script here to retrieve the array
    # For demonstration, let's assume you have a function "get_irradiance"
    # data_array = get_irradiance(lat, lon)  # Replace this with your actual function
    # return data_array
    pass


@app.callback(
    Output("text-field", "children"),
    Input("save-info-button", "n_clicks"),
    State("ppa-modal-price", "value"),
    State("ppa-modal-lat", "value"),
    State("ppa-modal-lon", "value"),
    prevent_initial_call=True
)
def save_ppa_info(n_clicks, ppa_price, ppa_lat, ppa_lon):
    # Save the information or perform any other action here
    info_text = f"PPA Price: {ppa_price}, Latitude: {ppa_lat}, Longitude: {ppa_lon}"
    return info_text


# @app.callback(
#     Output("ppa-modal", "is_open"),
#     Output("ppa-modal-lat", "value"),
#     Output("ppa-modal-lon", "value"),
#     Input("ppa-map", "click_lat_lng"),
#     prevent_initial_call=True
# )
# def open_ppa_modal_on_click(click_lat_lng):
#     if click_lat_lng:
#         lat, lon = click_lat_lng
#         return True, lat, lon
#     return dash.no_update, dash.no_update, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
