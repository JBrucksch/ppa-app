import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import math

def edit_series(profile):
    try:
        profile.index = pd.to_datetime(
            profile.index, format='%d-%m-%Y %H:%M:%S')
    except ValueError:
        profile.index = pd.to_datetime(
            profile.index, format='%Y-%m-%d %H:%M:%S')
    profile = profile.resample('1H').mean().interpolate('linear')

    # Cut the series to a maximum of 8760 datapoints (1 year)
    if len(profile) > 8760:
        profile = profile.iloc[:8760]

    profile.index = range(len(profile))

    return profile.values.flatten()


def create_demand_timeseries(demand):
    # Define load profile parameters
    peak_load_hour = 18  # Hour of the day with peak demand (0 to 23)
    peak_load_factor = 0.8  # Fraction of peak load compared to the maximum demand

    # Create hourly timeseries
    hours_per_day = 24
    days_per_year = 365
    total_demand = demand  # Total yearly demand (kWh)

    # Calculate hourly demand
    hourly_demand = np.zeros(hours_per_day)
    # hourly_demand[peak_load_hour] = total_demand * peak_load_factor
    # hourly_demand = hourly_demand * (total_demand / np.sum(hourly_demand))
    hourly_demand[:] = demand / 365 / 24

    # Repeat the hourly demand timeseries for the entire year
    yearly_demand_timeseries = np.tile(hourly_demand, days_per_year)

    #load csv file from input_data folder
    orig_hourly_demand = pd.read_csv('input_data\power_hourly_braeuer15.csv', header=None)

    # rescale so that the sum of the demand is the same as the total demand
    yearly_demand_timeseries = orig_hourly_demand * total_demand / orig_hourly_demand.sum()

    return yearly_demand_timeseries


def get_irradiance(lat, lon, size, tech):

    f = open("input_data\weather_data.pkl", "rb")
    weather_data = pickle.load(f)
    f.close()

    def calc_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = 6371 * c
        return distance

    filtered_dicts = [d for d in weather_data if d['tech'].lower() == tech.lower()]
    closest_dict = min(filtered_dicts, key=lambda x: calc_distance(lat, lon, float(x['lat']), float(x['lon'])))
    return np.array(closest_dict["values"])

    # todo find alternative publicly usable tools without renewables ninja, DWD
    # token = 'd6ec581bd3314680ee0b78b38c7dedc7e3f11a31'
    # api_base = 'https://www.renewables.ninja/api/'
    #
    # s = requests.session()
    # # Send token header with each request
    # s.headers = {'Authorization': 'Token ' + token}
    #
    # ##
    # # PV example
    # ##
    # if tech == "PV":
    #     url = api_base + 'data/pv'
    #     args = {
    #         'lat': lat,
    #         'lon': lon,
    #         'date_from': '2015-01-01',
    #         'date_to': '2015-12-31',
    #         'dataset': 'merra2',
    #         'capacity': size,
    #         'system_loss': 0.1,
    #         'tracking': 0,
    #         'tilt': 35,
    #         'azim': 180,
    #         'format': 'json'
    #     }
    # elif tech == "Wind":
    #     url = api_base + 'data/wind'
    #     args = {
    #         'lat': lat,
    #         'lon': lon,
    #         'date_from': '2015-01-01',
    #         'date_to': '2015-12-31',
    #         # 'dataset': 'merra2',
    #         'capacity': size,
    #         'height': 100,
    #         'turbine': 'Vestas V80 2000',
    #         'format': 'json'
    #     }
    #
    # r = s.get(url, params=args)
    #
    # # Parse JSON to get a pandas.DataFrame of data and dict of metadata
    # parsed_response = json.loads(r.text)
    #
    # data = pd.read_json(json.dumps(parsed_response['data']), orient='index')
    # # power = data * size
    # metadata = parsed_response['metadata']
    #
    # data.index = range(len(data))
    # return data.values.flatten()


def build_model_storage(model, demand, bat_size, agg_power_profile, var_dict, input_dict, price_profile, agg_price):
    ####################################################################################################################
    # Variables
    ####################################################################################################################

    # Decision Variables
    var_dict['p_grid_comp'] = pyo.Var(model.T, bounds=(0, None))  # Electricity bought from the grid
    model.add_component('p_grid_comp', var_dict['p_grid_comp'])

    var_dict['p_comp_grid'] = pyo.Var(model.T, bounds=(0, None))  # Electricity bought from the grid
    model.add_component('p_comp_grid', var_dict['p_comp_grid'])

    # Electricity bought with a baseload PPA
    var_dict['p_ppa_comp'] = pyo.Var(model.T, bounds=(0, None))
    model.add_component('p_ppa_comp', var_dict['p_ppa_comp'])

    var_dict['p_bat_load'] = pyo.Var(model.T, bounds=(0, None))  # Electricity from battery to load
    model.add_component('p_bat_load', var_dict['p_bat_load'])  # ToDo: kick out?

    var_dict['p_bat_comp'] = pyo.Var(model.T, bounds=(0, None))  # Electricity from battery to comp
    model.add_component('p_bat_comp', var_dict['p_bat_comp'])

    var_dict['p_comp_bat'] = pyo.Var(model.T, bounds=(0, None))  # Electricity from grid to battery
    model.add_component('p_comp_bat', var_dict['p_comp_bat'])

    var_dict['e_bat'] = pyo.Var(model.T_prime, bounds=(0, None))  # Energy in battery
    model.add_component('e_bat', var_dict['e_bat'])

    var_dict['p_comp_load'] = pyo.Var(model.T, bounds=(0, None))  # Electricity from grid to battery
    model.add_component('p_comp_load', var_dict['p_comp_load'])

    # bi-directional charging/discharging of grid and battery between the company load
    var_dict['z_bat'] = pyo.Var(model.T, domain=pyo.Binary)
    model.add_component('z_bat', var_dict['z_bat'])
    var_dict['z_grid'] = pyo.Var(model.T, domain=pyo.Binary)
    model.add_component('z_grid', var_dict['z_grid'])

    #######################################################################################################################
    # Constraints
    #######################################################################################################################

    # Constraints conservation company node
    # this constraint ensures that the total power consumed by the company at each time period equals
    # the total power supplied to the company, thereby ensuring energy conservation at the company node
    def rule(model, t):
        return var_dict['p_comp_load'][t] + var_dict['p_comp_bat'][t] + var_dict['p_comp_grid'][t] == \
               var_dict['p_grid_comp'][t] + var_dict['p_ppa_comp'][t] + var_dict['p_bat_comp'][t]

    model.add_component('Energy conservation company node', pyo.Constraint(model.T, rule=rule))

    # this constraint ensures that the energy stored in the battery at the end of each time period is equal to
    # the energy added minus the energy taken out, accounting for efficiency losses during charging and discharging
    def rule(model, t):
        return var_dict['e_bat'][t - 1] + var_dict['p_comp_bat'][t] * 0.95 - var_dict['p_bat_comp'][t] * (1 / 0.9) \
               == var_dict['e_bat'][t]

    model.add_component('Energy conservation battery', pyo.Constraint(model.T, rule=rule))

    # constraints general
    # this constraint ensures that the power purchased through the PPA matches the power generated by the PV system
    def rule(model, t):
        return var_dict['p_ppa_comp'][t] == agg_power_profile[t]

    model.add_component('PV profile', pyo.Constraint(model.T, rule=rule))

    # this constraint ensures that the power consumed by loads matches the demand at each time period
    def rule(model, t):
        return var_dict['p_comp_load'][t] == demand[t]

    model.add_component('Demand matching', pyo.Constraint(model.T, rule=rule))

    def rule(model, t):
        return var_dict['e_bat'][t] <= bat_size

    model.add_component('Battery size', pyo.Constraint(model.T, rule=rule))

    # this constraint ensures that the power charging rate of the battery does not exceed a certain limit, which is determined by the battery size
    # 1 is the placeholder for the e2p ratio of the battery, e2p ratio tells how much charging or discharging power
    # can we draw from the battery per installed capacity(energy)
    def rule(model, t):
        return var_dict['p_comp_bat'][t] <= bat_size / 1  # WHY divide by 1

    model.add_component('Power limit CHA', pyo.Constraint(model.T, rule=rule))

    # same for discharging
    def rule(model, t):
        return var_dict['p_bat_comp'][t] <= bat_size / 1

    model.add_component('Power limit DCH', pyo.Constraint(model.T, rule=rule))

    # this constraint ensures that the initial energy level of the battery is zero
    model.add_component('Battery init energy', pyo.Constraint(expr=model.e_bat[-1] == 0))

    # bi-directional charging/discharging constraints of grid and battery between the company load
    bigM = 1e9

    # This constraint limits the power purchased from the grid based on the binary variable z_grid
    # If z_grid is activated(set to 1), indicating that the grid is connected, the power purchased from the grid
    # is constrained to be less than or equal to bigM.
    # If z_grid is not activated(set to 0), the constraint effectively becomes inactive, as bigM * 0 = 0
    def rule(model, t):
        return var_dict['p_grid_comp'][t] <= var_dict['z_grid'][t] * bigM

    model.add_component('Bidirectional constraint grid 1', pyo.Constraint(model.T, rule=rule))

    # This constraint limits the power sold to the grid, based on the complement of binary variable z_grid
    def rule(model, t):
        return var_dict['p_comp_grid'][t] <= (1 - var_dict['z_grid'][t]) * bigM

    model.add_component('Bidirectional constraint grid 2', pyo.Constraint(model.T, rule=rule))

    # This constraint limits the power charging the battery
    def rule(model, t):
        return var_dict['p_comp_bat'][t] <= var_dict['z_bat'][t] * bigM

    model.add_component('Bidirectional constraint battery 1', pyo.Constraint(model.T, rule=rule))

    # This constraint limits the power discharging from the battery
    def rule(model, t):
        return var_dict['p_bat_comp'][t] <= (1 - var_dict['z_bat'][t]) * bigM

    model.add_component('Bidirectional constraint battery 2', pyo.Constraint(model.T, rule=rule))

    # PEAK SHAVING (LOAD SHEDDING)

    # Electricity bought from the grid
    # maximum power that can be bought from the grid
    var_dict['p_max_grid'] = pyo.Var(bounds=(0, None))
    model.add_component('p_max_grid', var_dict['p_max_grid'])

    # This constraint ensures that the power purchased from the grid (var_dict['p_max_grid']) is greater than or equal to
    # the sum of power obtained through the power purchase agreement (var_dict['p_ppa_comp'][t]),
    # power purchased directly from the grid (var_dict['p_grid_comp'][t]), and
    # the power supplied to the grid (var_dict['p_comp_grid'][t]) for each time period t.
    def rule(model, t):
        return var_dict['p_max_grid'] >= var_dict['p_ppa_comp'][t] + \
               var_dict['p_grid_comp'][t] - var_dict['p_comp_grid'][t]

    model.add_component('Max grid power', pyo.Constraint(model.T, rule=rule))

    # Objective Function: Minimize total electricity costs

    annual_factor = (365 * 24) / len(model.T)  # if the optimized time is only a share of the whole year

    model.objective = pyo.Objective(
        expr=(
                pyo.quicksum(input_dict['price_grid'] * var_dict['p_grid_comp'][t] -
                             price_profile[t] * var_dict['p_comp_grid'][t] for t in model.T) * annual_factor
                + var_dict['p_max_grid'] * input_dict['price_peak_load']
        ),
        sense=pyo.minimize
    )


def calculate_annuity(initial_investment, interest_rate, investment_duration):
    annuity = (initial_investment * interest_rate) / \
              (1 - (1 + interest_rate) ** -investment_duration)
    return annuity


def extract_results(var_dict, time_steps):
    print("Extracting results...")
    num_rows = len(time_steps)
    num_cols = len(var_dict)

    result_matrix = np.zeros((num_rows, num_cols))
    for j, var in enumerate(var_dict.values()):
        if isinstance(var, pyo.Var):
            var_name = var.getname()	# get the name of the variable
            if var.is_indexed():
                var_data = np.array(list(var.get_values().values()))
                if len(var_data) == num_rows:
                    result_matrix[:, j] = var_data
                else:
                    result_matrix[:,j] = var_data[1:]
            else:
                result_matrix[:, j] = pyo.value(var)
        else:
            for i, t in enumerate(time_steps):
                result_matrix[i, j] = 0

    result_df = pd.DataFrame(
        result_matrix, index=time_steps, columns=list(var_dict.keys()))

    return result_df


def get_optimization_result(input_dict, t_horizon, pp_data, load_profile, ps_boolean):
    
    power_profiles = []
    energy_from_ppa_df = pd.DataFrame(columns=['pv', 'wind'], index=range(t_horizon), data=0)
    costs_from_ppa_df = pd.DataFrame(columns=['pv', 'wind'], index=range(t_horizon), data=0)
    generator_profiles_df = pd.DataFrame(index=range(t_horizon))
    generator_prices_df = pd.DataFrame(index=range(t_horizon))
    for key, power_profile in pp_data.items():
        plant_size = power_profile[1] * 1000 # in kW
        power_factors = power_profile[5]
        contract_price = power_profile[0]
        power_profiles.append([x * plant_size for x in power_factors])
        generator_profiles_df[key] = pd.array([x * plant_size for x in power_factors])
        generator_prices_df[key] = contract_price 

        if power_profile[2] == "PV":
            energy_from_ppa_df['pv'] += [x * plant_size for x in power_factors]
            costs_from_ppa_df['pv'] += [x * plant_size * contract_price for x in power_factors]
        elif power_profile[2] == "Wind":
            energy_from_ppa_df['wind'] += [x * plant_size for x in power_factors]
            costs_from_ppa_df['wind'] += [x * plant_size * contract_price for x in power_factors]
    agg_power_profile = [sum(x) for x in zip(*power_profiles)]

    for key, power_profile in pp_data.items():
        plant_size = power_profile[1] * 1000 # in kW
        power_factors = power_profile[5]
        pp_data[key].append([x * plant_size for x in power_factors])
        #power_profiles.append([x * power_profile[1] * 1000 for x in power_profile[5]])

    agg_price = []
    for key, power_profile in pp_data.items():
        price = power_profile[0]
        generated_elec = power_profile[6]
        agg_price.append([x * price for x in generated_elec])
    agg_price = [sum(x) for x in zip(*agg_price)]
    
    opt = False


    ### OPTIMIZATION ######

    var_dict = {}

    ### SIMULATION ######

    # Battery specs
    battery_capacity = input_dict['battery_capacity']  # in kWh
    battery_charge_rate = input_dict['battery_capacity'] * input_dict['crate_bat']  # Max kW it can charge per hour
    battery_discharge_rate = input_dict['battery_capacity'] * input_dict['crate_bat']  # Max kW it can discharge per hour
    battery_storage = 0  # Initial state of charge
    battery_eff = input_dict['efficiency_bat']  # 1% loss per kwh discarge or charge

    ps_lower_limit = 0.2 * battery_capacity
    ps_upper_limit = 0.8 * battery_capacity
    
    # Initialize result columns
    net_grid_interaction = pd.Series(np.zeros(t_horizon))  # Positive for buying, negative for selling
    battery_activity = pd.Series(np.zeros(t_horizon))  # Positive for charging, negative for discharging
    
    peak_limit = input_dict['max_peak_power']  # in kW
    # Simulation loop
    for hour in range(t_horizon):
        demand_to_serve = load_profile[hour]
        if demand_to_serve > peak_limit and battery_storage > 0 and ps_boolean:
            print("Peak shaving in hour", hour)
            shaving_demand = load_profile[hour] - peak_limit
            discharge_amount = min(battery_discharge_rate, battery_storage, shaving_demand)
            battery_storage -= discharge_amount
            demand_to_serve -= discharge_amount
            battery_activity[hour] = -discharge_amount

        total_generation = generator_profiles_df.iloc[hour].sum()
        net_demand = demand_to_serve - total_generation

        if net_demand > 0:  # Need more power
            if battery_storage > ps_lower_limit:  # Battery has charge
                discharge_amount = min(battery_discharge_rate, battery_storage - ps_lower_limit, net_demand)
                battery_storage -= discharge_amount
                net_demand -= discharge_amount
                battery_activity[hour] = -discharge_amount
            if net_demand > 0:  # Still need more power
                net_grid_interaction[hour] = net_demand
        else:  # Excess power
            excess_power = -net_demand
            for gen in generator_prices_df.columns:
                if input_dict['injection_price'] > generator_prices_df.at[hour, gen]:  # Sell if market price > generation cost
                    sell_amount = power_profiles.at[hour, gen]
                    net_grid_interaction[hour] -= sell_amount  # Negative for selling
                    excess_power -= sell_amount
            
            if excess_power > 0 and battery_storage < battery_capacity and ps_boolean:  # Remaining excess goes to battery
                charge_amount = min(battery_charge_rate, battery_capacity - battery_storage, excess_power, peak_limit - load_profile[hour])
                battery_storage += charge_amount
                excess_power -= charge_amount
                battery_activity[hour] = charge_amount
            elif excess_power > 0 and battery_storage < battery_capacity:  # Remaining excess goes to battery
                charge_amount = min(battery_charge_rate, battery_capacity - battery_storage, excess_power)
                battery_storage += charge_amount
                excess_power -= charge_amount
                battery_activity[hour] = charge_amount
            if excess_power > 0:  # Still excess power
                net_grid_interaction[hour] -= excess_power

    ### RESULTS ######

    data_dict = {
        'annuity': 0,
        'inv_costs': 0,
        'elec_costs_grid': 0,
        'elec_costs_ppa_pv': 0,
        'elec_costs_ppa_wind': 0,
        'elec_revenue': 0,
        'peak_power_costs': 0,
        'emissions': 0,
        'energy_from_grid': 0,
        'energy_to_grid': 0,
        'energy_from_ppa': 0,
        'energy_to_battery': 0,
        'energy_from_battery': 0,
        'energy_to_company': 0,
        'timeseries': pd.DataFrame()
    }

    result_dict = {'Storage': data_dict.copy(),
                   'PPA': data_dict.copy(),
                   'Grid': data_dict.copy()}

    storage_dict = result_dict['Storage']
    #storage_dict['timeseries'] = extract_results(var_dict, list(range(t_horizon)))
    energy_from_grid_ts = np.array([x if x > 0 else 0 for x in net_grid_interaction])
    energy_to_grid_ts = np.array([-1*x if x < 0 else 0 for x in net_grid_interaction])
    energy_from_ppa_ts = np.array(agg_power_profile)
    energy_to_battery_ts = np.array([x if x > 0 else 0 for x in battery_activity])
    energy_from_battery_ts = np.array([-1*x if x < 0 else 0 for x in battery_activity])

    to_comp_energy = energy_to_battery_ts + load_profile - energy_from_battery_ts
    peak_power = max(to_comp_energy)
    peak_power_costs = peak_power * input_dict['price_peak_load']

    energy_from_grid = energy_from_grid_ts.sum()
    energy_to_grid = energy_to_grid_ts.sum()
    energy_from_ppa = energy_from_ppa_ts.sum()
    energy_to_battery = energy_to_battery_ts.sum()
    energy_from_battery = energy_from_battery_ts.sum()

    blc_node = energy_from_grid - energy_to_grid + energy_from_ppa - energy_to_battery + energy_from_battery - load_profile.sum()

    df_to_save = pd.DataFrame({'load': load_profile, 'pv': energy_from_ppa_df['pv'], 'wind': energy_from_ppa_df['wind'], 
                               'to_bat': energy_to_battery_ts, 'from_bat': energy_from_battery_ts, 'to_grid': energy_to_grid_ts})
    # save that df
    df_to_save.to_excel('results.xlsx')

    result_storage = pd.DataFrame({'p_ppa_comp': agg_power_profile, 'p_comp_load': load_profile.tolist()})

    need_for_grid = load_profile - agg_power_profile
    result_storage['p_grid_comp'] = energy_from_grid_ts
    need_for_injection = agg_power_profile - load_profile
    result_storage['p_comp_grid'] = energy_to_grid_ts
    result_storage['p_bat_comp'] = energy_from_battery_ts
    result_storage['p_comp_bat'] = energy_to_battery_ts

    storage_dict['inv_costs'] = calculate_annuity(
                input_dict['battery_capacity'] * input_dict['price_bat'], input_dict['interest_rate'],
                input_dict['investment_horizon'])
    storage_dict['elec_costs_grid'] = input_dict['price_grid'] * energy_from_grid  
    storage_dict['elec_revenue'] = sum(input_dict['injection_price'] * energy_to_grid_ts) 
    storage_dict['elec_costs_ppa_pv'] = costs_from_ppa_df['pv'].sum()  # in €
    storage_dict['elec_costs_ppa_wind'] = costs_from_ppa_df['wind'].sum()  # in €
    storage_dict['peak_power_costs'] = peak_power_costs  # in €
    storage_dict['annuity'] = storage_dict['inv_costs'] + storage_dict['elec_costs_grid'] - storage_dict['elec_revenue'] +\
                                storage_dict['elec_costs_ppa_pv'] + storage_dict['elec_costs_ppa_wind'] + storage_dict['peak_power_costs']	# in €

    storage_dict['emissions'] = energy_from_grid / 1000 * 0.380  # in t
    storage_dict['energy_from_grid'] = energy_from_grid / 1000  # in MWh
    storage_dict['energy_to_grid'] = energy_to_grid / 1000  # in MWh
    storage_dict['energy_from_ppa_pv'] = energy_from_ppa_df['pv'].sum() / 1000  # in MWh
    storage_dict['energy_from_ppa_wind'] = energy_from_ppa_df['wind'].sum() / 1000  # in MWh
    storage_dict['energy_to_company'] = (energy_from_ppa + energy_from_grid - energy_to_grid) / 1000  # in MWh

    storage_dict['energy_to_battery'] = energy_to_battery / 1000  # in MWh
    storage_dict['energy_from_battery'] = energy_from_battery / 1000  # in MWh
    storage_dict['cycles_battery'] = energy_to_battery / input_dict['battery_capacity']  # number of cycles

    storage_dict['timeseries'] = result_storage
    
    print("Optimized Electricity Purchase from Storage:")
    print('Total Cost Storage:' + str(storage_dict['annuity']))

    # PPA --------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    result_ppa_pap = pd.DataFrame({'p_ppa_comp': agg_power_profile, 'p_comp_load': load_profile.tolist()})

    need_for_grid = load_profile - agg_power_profile
    result_ppa_pap['p_grid_comp'] = np.where(agg_power_profile < load_profile, need_for_grid.T, 0)
    need_for_injection = agg_power_profile - load_profile
    result_ppa_pap['p_comp_grid'] = np.where(agg_power_profile > load_profile, need_for_injection.T, 0)

    total_demand = result_ppa_pap['p_grid_comp'] + result_ppa_pap['p_ppa_comp'] - result_ppa_pap['p_comp_grid']

    energy_from_grid = result_ppa_pap['p_grid_comp'].sum()
    energy_to_grid = result_ppa_pap['p_comp_grid'].sum()


    result_dict['PPA']['inv_costs'] = 0
    result_dict['PPA']['elec_costs_grid'] = input_dict['price_grid'] * energy_from_grid
    result_dict['PPA']['elec_costs_ppa_pv'] = costs_from_ppa_df['pv'].sum()
    result_dict['PPA']['elec_costs_ppa_wind'] = costs_from_ppa_df['wind'].sum()
    result_dict['PPA']['elec_revenue'] = sum(input_dict['injection_price'] * result_ppa_pap['p_comp_grid'])
    result_dict['PPA']['peak_power_costs'] = input_dict['price_peak_load'] * total_demand.max()
    result_dict['PPA']['annuity'] = result_dict['PPA']['inv_costs'] + result_dict['PPA']['elec_costs_grid'] - \
                                    result_dict['PPA']['elec_revenue'] + result_dict['PPA']['elec_costs_ppa_pv'] + \
                                    result_dict['PPA']['elec_costs_ppa_wind'] + result_dict['PPA']['peak_power_costs']
    
    result_dict['PPA']['emissions'] = energy_from_grid / 1000 * 0.380
    result_dict['PPA']['energy_from_grid'] = energy_from_grid / 1000
    result_dict['PPA']['energy_to_grid'] = energy_to_grid / 1000
    result_dict['PPA']['energy_from_ppa_pv'] = energy_from_ppa_df['pv'].sum() / 1000
    result_dict['PPA']['energy_from_ppa_wind'] = energy_from_ppa_df['wind'].sum() / 1000
    result_dict['PPA']['energy_to_company'] = (total_demand) / 1000

    result_dict['PPA']['timeseries'] = result_ppa_pap

    print("Optimized Electricity Purchase with only PPA PAP:")
    print(f"Total Annuity PAP: {result_dict['PPA']['annuity']}")

    # only grid ------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------

    result_grid = pd.DataFrame({'p_comp_load': load_profile.tolist()})
    result_grid['p_comp_grid'] = 0
    result_grid['p_grid_comp'] = load_profile

    total_demand = result_grid['p_grid_comp'] - result_grid['p_comp_grid']

    energy_from_grid_grid = result_grid['p_grid_comp'].sum()
    energy_to_grid_grid = result_grid['p_comp_grid'].sum()

    
    result_dict['Grid']['inv_costs'] = 0
    result_dict['Grid']['elec_costs_grid'] = input_dict['price_grid'] * energy_from_grid_grid
    result_dict['Grid']['elec_costs_ppa_pv'] = 0
    result_dict['Grid']['elec_costs_ppa_wind'] = 0
    result_dict['Grid']['elec_revenue'] = sum(input_dict['injection_price'] * result_grid['p_comp_grid'])
    result_dict['Grid']['peak_power_costs'] = input_dict['price_peak_load'] * total_demand.max()
    result_dict['Grid']['annuity'] = result_dict['Grid']['inv_costs'] + result_dict['Grid']['elec_costs_grid'] - \
                                    result_dict['Grid']['elec_revenue'] + result_dict['Grid']['elec_costs_ppa_pv'] + \
                                    result_dict['Grid']['elec_costs_ppa_wind'] + result_dict['Grid']['peak_power_costs']    
    
    result_dict['Grid']['emissions'] = energy_from_grid_grid / 1000 * 0.380
    result_dict['Grid']['energy_from_grid'] = energy_from_grid_grid / 1000
    result_dict['Grid']['energy_to_grid'] = energy_to_grid_grid / 1000
    result_dict['Grid']['energy_from_ppa_pv'] = 0
    result_dict['Grid']['energy_from_ppa_wind'] = 0
    result_dict['Grid']['energy_to_company'] = (total_demand) / 1000

    result_dict['Grid']['timeseries'] = result_grid

    print("Optimized Electricity Purchase from Grid:")
    print(f"Total Cost Grid: {result_dict['Grid']['annuity']}")

    return result_dict


def plot_cost_analy(result_dict, input_dict):
    results_storage = result_dict['Storage']
    results_ppa = result_dict['PPA']
    results_grid = result_dict['Grid']

    # Prepare your plot
    fig_cost_anls = go.Figure()

    categories = ["Grid", "incl. PPA", "incl. PPA + Storage"]
    total_values = [0] * 3  # List to store total values for each category

    # Adding each bar and calculating totals
    bar_data = [
        {'x': categories, 'y': [0, 0, results_storage['inv_costs']*-1], 'name': "Costs Battery Investment", 'marker_color': 'rgba(142, 186, 229, 1)'},
        {'x': categories, 'y': [results_grid['elec_costs_grid']*-1, results_ppa['elec_costs_grid']*-1, results_storage['elec_costs_grid']*-1], 'name': "Costs Supplier", 'marker_color': 'rgba(0, 84, 159, 1)'},
        {'x': categories, 'y': [0, results_ppa['elec_costs_ppa_pv']*-1, results_storage['elec_costs_ppa_pv']*-1], 'name': "Costs PPA PV", 'marker_color': 'rgba(255, 237, 0, 1)'},
        {'x': categories, 'y': [0, results_ppa['elec_costs_ppa_wind']*-1, results_storage['elec_costs_ppa_wind']*-1], 'name': "Costs PPA Wind", 'marker_color': 'rgba(87, 171, 89, 1)'},
        {'x': categories, 'y': [results_grid['peak_power_costs']*-1, results_ppa['peak_power_costs']*-1, results_storage['peak_power_costs']*-1], 'name': "Costs Peak Power", 'marker_color': 'rgba(227, 0, 102, 1)'},
        {'x': categories, 'y': [0, results_ppa['elec_revenue'], results_storage['elec_revenue']], 'name': "Revenue Market", 'marker_color': 'rgba(0, 152, 161, 1)'}
    ]

    for data in bar_data:
        fig_cost_anls.add_trace(go.Bar(x=data['x'], y=data['y'], name=data['name'], marker_color=data['marker_color']))
        total_values = [sum(x) for x in zip(total_values, data['y'])]

    # Update the layout
    fig_cost_anls.update_layout(
        yaxis_title="Yearly Revnue/Costs in €",
        yaxis=dict(title_font=dict(size=14)),  # Modify the range as per your requirement
        xaxis=dict(tickfont=dict(size=14)),
        barmode='relative',
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    # Adding text labels for totals
    # for i, value in enumerate(total_values):
    #     fig_cost_anls.add_annotation(x=categories[i], y=bar_data[5]['y'][i] + 1000,  # Adjust position offset as needed
    #                                 text=f"{int(value/1000)} k€",
    #                                 showarrow=False,
    #                                 font=dict(color="black", size=16))

    # fig_cost_anls.show()
    return fig_cost_anls


def plot_load_dur_curve(model_storage):
    p_ppa_comp = [model_storage.p_ppa_comp[index].value for index in model_storage.p_ppa_comp]
    p_grid_comp = [model_storage.p_grid_comp[index].value for index in model_storage.p_grid_comp]
    p_comp_grid = [model_storage.p_comp_grid[index].value for index in model_storage.p_comp_grid]
    p_comp_bat = [model_storage.p_comp_bat[index].value for index in model_storage.p_comp_bat]

    sorted_ppa_comp = sorted(p_ppa_comp, reverse=True)
    sorted_grid_comp = sorted(p_grid_comp, reverse=True)
    sorted_comp_grid = sorted(p_comp_grid, reverse=True)
    sorted_comp_bat = sorted(p_comp_bat, reverse=True)
    cumulated_duration = list(range(1, len(sorted_ppa_comp) + 1))

    fig_load_duration = make_subplots(specs=[[{"secondary_y": True}]])
    fig_load_duration.add_trace(go.Scatter(x=cumulated_duration, y=sorted_ppa_comp,
                                           mode="lines", name="PPA to Company", marker_color="green"))
    fig_load_duration.add_trace(go.Scatter(x=cumulated_duration, y=sorted_grid_comp,
                                           mode="lines", name="Grid to Company", marker_color="blue"))
    fig_load_duration.add_trace(go.Scatter(x=cumulated_duration, y=sorted_comp_grid,
                                           mode="lines", name="Company to Grid", marker_color="red"))
    fig_load_duration.add_trace(go.Scatter(x=cumulated_duration, y=sorted_comp_bat,
                                           mode="lines", name="Company to Battery", marker_color="orange"))
    fig_load_duration.update_layout(title='Load Duration Curve',
                                    xaxis_title='Cumulated Duration (hours)',
                                    yaxis_title='Energy (kWh)')
    return fig_load_duration


def plot_elec_generated(pp_data, ppa_size):

    for key, power_profile in pp_data.items():
        pp_data[key][5] = [x * ppa_size * 1000 for x in power_profile[5]]

    monthly_power = {"PV": [0.0] * 12, "Wind": [0.0] * 12}
    pv_keys = []
    wind_keys = []

    for pp in pp_data.values():
        sums = []
        for i in range(0, len(pp[5]), 730):  # on avg each month has 730 hrs
            chunk_sum = sum(pp[5][i:i + 730])
            sums.append(chunk_sum)
        if pp[2] == "PV":
            monthly_power["PV"] = [x + y for x, y in zip(monthly_power["PV"], sums)]
        elif pp[2] == "Wind":
            monthly_power["Wind"] = [x + y for x, y in zip(monthly_power["Wind"], sums)]

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    PV_bar = go.Bar(x=months, y=monthly_power["PV"], name='PV', marker_color='#FFFF00')
    wind_bar = go.Bar(x=months, y=monthly_power["Wind"], name='Wind', marker_color='#00FF00')
    fig_pp_gentn = go.Figure(data=[PV_bar, wind_bar])
    fig_pp_gentn.update_layout(title='Power Production', xaxis_title='Months', yaxis_title='Energy (kWh)')

    return fig_pp_gentn


def plot_individual_prod(pp_data, ppa_size):
    for key, power_profile in pp_data.items():
        pp_data[key][5] = [x * ppa_size * 1000 for x in power_profile[5]]

    monthly_power = {"PV": [], "Wind": []}
    pv_keys = []
    wind_keys = []

    for pp_key, pp_val in pp_data.items():
        sums = []
        for i in range(0, len(pp_val[5]), 730):  # on avg each month has 730 hrs
            chunk_sum = sum(pp_val[5][i:i + 730])
            sums.append(chunk_sum)
        if pp_val[2] == "PV":
            monthly_power["PV"].append(sums)
            pv_keys.append(pp_key)
        elif pp_val[2] == "Wind":
            monthly_power["Wind"].append(sums)
            wind_keys.append(pp_key)

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']

    shades_of_yellow = ['#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FFD700',
                        '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FFFF00']
    pv_traces = []
    for i, key in enumerate(pv_keys):
        pv_bar = go.Bar(x=months, y=monthly_power["PV"][i], name=key, marker_color=shades_of_yellow[i])
        pv_traces.append(pv_bar)
    fig_pv = go.Figure(data=pv_traces)
    fig_pv.update_layout(title='PV Power Production', xaxis_title='Months', yaxis_title='Energy (kWh)',
                         barmode='stack')

    shades_of_green = ['#00FF00', '#7FFF00', '#ADFF2F', '#9ACD32', '#556B2F',
                       '#228B22', '#008000', '#006400', '#32CD32', '#90EE90']
    wind_traces = []
    for i, key in enumerate(wind_keys):
        wind_bar = go.Bar(x=months, y=monthly_power["Wind"][i], name=key, marker_color=shades_of_green[i])
        wind_traces.append(wind_bar)
    fig_wind = go.Figure(data=wind_traces)
    fig_wind.update_layout(title='Wind Power Production', xaxis_title='Months', yaxis_title='Energy (kWh)',
                           barmode='stack')

    return fig_pv, fig_wind


def plot_fig4(t_horizon, result_dict, load_profile, price_profile, agg_power_profile):
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=pd.Series(range(t_horizon)), y=result_dict["Storage"]["energy_from_grid_all"],
                              mode="lines", name="Storage", marker_color="green"))
    fig4.add_trace(go.Scatter(x=pd.Series(range(t_horizon)), y=result_dict["PPA"]["energy_from_grid_all"],
                              mode="lines", name="PPA", marker_color="blue"))
    fig4.add_trace(
        go.Scatter(x=pd.Series(range(t_horizon)), y=load_profile, mode="lines", name="Grid", marker_color="grey"))
    fig4.add_trace(go.Scatter(x=pd.Series(range(t_horizon)), y=price_profile, mode="lines",
                              name="Wholesale price 2023", marker_color="red"), secondary_y=True)
    fig4.add_trace(go.Scatter(x=pd.Series(range(t_horizon)), y=agg_power_profile, mode="lines",
                              name="Aggregated Generation", marker_color="orange"))
    fig4.update_layout(title='Energy Comparison Over Time',
                       xaxis_title='Time Horizon',
                       yaxis_title='Energy (kWh)',
                       yaxis2_title='Price (Euro/kWh)')
    return fig4


def plot_emissions(result_dict):

    emsn_grid = (result_dict["Grid"]["energy_from_grid"] - result_dict["Grid"]["energy_to_grid"]) * 0.00042
    emsn_grid_ppa = (result_dict["PPA"]["energy_from_grid"] - result_dict["PPA"]["energy_to_grid"]) * 0.00042
    emsn_grid_ppa_bat = (result_dict["Storage"]["energy_from_grid"] - result_dict["Storage"]["energy_to_grid"]) * 0.00042

    values = [emsn_grid, emsn_grid_ppa, emsn_grid_ppa_bat]
    categories = ["Grid", "Grid + PPA", "Grid + PPA + Battery"]
    colours = ['#00549F', '#407FB7', '#8EBAE5']

    fig_emsn = go.Figure(data=[go.Bar(x=categories, y=values, marker_color=colours)])
    fig_emsn.update_layout(title="Emmisions",
                           xaxis_title="Scenarios",
                           yaxis_title="CO2 in Tons"
                           )

    return fig_emsn


def plot_consmp_dmnd(result_dict, load_profile):

    energy_from_grid_plus_ppas = result_dict["Storage"]["energy_from_grid_all"] + result_dict["Storage"][
        "energy_from_ppa_all"]

    energy_from_grid_plus_ppas = np.reshape(energy_from_grid_plus_ppas, (12, -1))
    load_profile = np.reshape(load_profile, (12, -1))

    # compute the monthly sum
    energy_from_grid_plus_ppas = np.sum(energy_from_grid_plus_ppas, axis=1)
    load_profile = np.sum(load_profile, axis=1)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    energy_from_grid_plus_ppas_df = pd.DataFrame({'Month': months, 'Grid + PPAs': energy_from_grid_plus_ppas})
    load_profile_df = pd.DataFrame({'Month': months, 'Demand': load_profile})
    df_merged = pd.merge(energy_from_grid_plus_ppas_df, load_profile_df, on='Month')

    fig = go.Figure(data=[
        go.Bar(name="Grid + PPAs", x=df_merged["Month"], y=df_merged["Grid + PPAs"], marker=dict(color='#FFA500')),
        go.Bar(name='Demand', x=df_merged["Month"], y=df_merged["Demand"], marker=dict(color='#407FB7'))
    ])

    fig.update_layout(barmode='group', title='Consumtion and Demand', xaxis_title='Month', yaxis_title='Energy (kWh)')
    return fig


def plot_donut(result_dict, load_profile, agg_power_profile, pp_data):

    figs_donut_autarky = []
    figs_donut_imports = []
    figs_donut_ppa_usage = []

    for scn_name, scn in result_dict.items():
        if scn_name == 'Storage':
            title_str = "PPA + Storage"
        elif scn_name == 'PPA':
            title_str = "PPA"
        elif scn_name == 'Grid':
            title_str = "Only Grid"
        data = result_dict[scn_name]

        # get aggregated profiles for pv and wind
        wind_profile = np.zeros(len(load_profile))
        pv_profile = np.zeros(len(load_profile))
        # find the cheapest power plant in pp_data
        plant_numbers_sorted_for_price = []
        for key, plant_data in pp_data.items():
            power_profile = plant_data[6]
            plant_price = plant_data[0]
            
            if len(plant_numbers_sorted_for_price) == 0:
                plant_numbers_sorted_for_price.append((plant_price, key, plant_data[2], power_profile))
            for i in range(len(plant_numbers_sorted_for_price)):
                if plant_price < plant_numbers_sorted_for_price[i][0]:
                    plant_numbers_sorted_for_price.insert(i, (plant_price, key, plant_data[2], power_profile))
                    break
                if i == len(plant_numbers_sorted_for_price) - 1:
                    plant_numbers_sorted_for_price.append((plant_price, key, plant_data[2], power_profile))
                    break

            if plant_data[2] == "PV":
                pv_profile += power_profile
            elif plant_data[2] == "Wind":
                wind_profile += power_profile

        # get ts for grid import
        grid_import_ts = np.array(data['timeseries']['p_grid_comp'])
        grid_export_ts = np.array(data['timeseries']['p_comp_grid'])

        timesteps_with_both_values_larger_than_one = np.where((grid_import_ts > 0) & (grid_export_ts > 0))[0]
        if len(timesteps_with_both_values_larger_than_one) > 0:
            print("There are timesteps where both grid import and export values are larger than one.")
        else:
            print("There are no timesteps where both grid import and export values are larger than one.")

        #company xport
        if scn_name == 'Storage':
            company_export_ts = data['timeseries']['p_bat_comp'] - load_profile
            company_export_ts = np.where(company_export_ts < 0, 0, company_export_ts)
        else:
            company_export_ts = np.zeros(len(load_profile))
        company_export_sum = company_export_ts.sum()
        print("Company export sum in scenario ", scn_name, ": ", company_export_sum)

        # calculate the moment where wind power is sold to the grid
        wind_export_ts = np.zeros(len(grid_export_ts))
        pv_export_ts = np.zeros(len(grid_export_ts))
        for t in range(len(grid_export_ts)):
            remaining_export = grid_export_ts[t] - company_export_ts[t]
            if remaining_export > 0:
                # find plant ith lowest price
                for plant_info in plant_numbers_sorted_for_price:
                    power_value = plant_info[3][t]
                    if plant_info[2] == "PV":
                        pv_export_ts[t] += min(remaining_export, power_value)
                    elif plant_info[2] == "Wind":
                        wind_export_ts[t] += min(remaining_export, power_value)
                    remaining_export -= min(remaining_export, power_value)
                    if remaining_export <= 0:
                        break
        
        sum_wind_and_pv_exports = wind_export_ts.sum() + pv_export_ts.sum()

        if sum_wind_and_pv_exports == data['timeseries']['p_comp_grid'].sum()-company_export_sum:
            print("Export validation check.")
        else:
            print("Export validation check failed.")

        wind_power_used_ts = wind_profile - wind_export_ts
        pv_power_used_ts = pv_profile - pv_export_ts

        ############################################################################################################
        # Energy Sources
        ############################################################################################################

        energy_from_grid = data['energy_from_grid']
        energy_from_ppa_pv = data['energy_from_ppa_pv'] 
        energy_from_ppa_wind = data['energy_from_ppa_wind']
        energy_from_ppa = energy_from_ppa_pv + energy_from_ppa_wind
        ppa_supply_rate = energy_from_ppa / (energy_from_ppa + energy_from_grid) * 100

        fig_donut = go.Figure(data=[go.Pie(labels=['Grid', 'PV PPA', 'Wind PPA'],
                        values=[energy_from_grid, energy_from_ppa_pv, energy_from_ppa_wind],
                        hole=0.3,
                        marker=dict(colors=['rgba(0, 84, 159, 1)', 'rgba(255, 237, 0, 1)', 'rgba(87, 171, 89, 1)']),
                        textinfo='label+percent',
                        title=str(int(ppa_supply_rate)) + ' %')])
        fig_donut.update_layout(
            title=title_str,
            title_x=0.5,
            title_font=dict(
            size=20,
            color="black",
            family="Arial"
            )
        )
        figs_donut_imports.append(fig_donut)

        ############################################################################################################
        # Energy Usage
        ############################################################################################################

        energy_to_grid = data['energy_to_grid']
        energy_from_ppa_used = data['energy_from_ppa_pv'] + data['energy_from_ppa_wind'] - data['energy_to_grid']
        
        autarky_rate = energy_from_ppa_used / (energy_from_ppa_used + energy_from_grid) * 100
        fig_donut = go.Figure(data=[go.Pie(labels=['Grid', 'PPA'],
                        values=[energy_from_grid, energy_from_ppa_used],
                        hole=0.3,
                        marker=dict(colors=['rgba(0, 84, 159, 1)', 'rgba(142, 186, 229, 1)']),
                        textinfo='label+percent',
                        title=str(int(autarky_rate)) + ' %')])
        fig_donut.update_layout(
            title=title_str,
            title_x=0.5,
            title_font=dict(
            size=20,
            color="black",
            family="Arial"
            )
        )
        figs_donut_autarky.append(fig_donut)

        ############################################################################################################
        # PPA Usage
        ############################################################################################################
        if scn_name == 'Grid':
            continue
        ppa_used_pv = pv_power_used_ts.sum()/1000
        ppa_used_wind = wind_power_used_ts.sum()/1000
        ppa_unused_pv = data['energy_from_ppa_pv'] - ppa_used_pv
        ppa_unused_wind = data['energy_from_ppa_wind'] - ppa_used_wind

        ppa_usage_rate = (ppa_used_pv + ppa_used_wind) / (ppa_used_pv + ppa_used_wind + ppa_unused_pv + ppa_unused_wind) * 100

        fig_donut = go.Figure(data=[go.Pie(labels=['PV used', 'Wind used', 'Wind sold', 'PV sold'],
                                        values=[ppa_used_pv,ppa_used_wind, ppa_unused_wind, ppa_unused_pv],
                                        hole=0.3,
                                        marker_colors=['rgba(255, 237, 0, 1)', 'rgba(87, 171, 89, 1)', 'rgba(184, 214, 152, 1)', 'rgba(255, 245,155, 1)'],
                                        textinfo='percent',
                                        title=str(int(ppa_usage_rate)) + ' %')])
        fig_donut.update_layout(
            title=title_str,
            title_x=0.5,
            title_font=dict(
            size=20,
            color="black",
            family="Arial")
        )

        figs_donut_ppa_usage.append(fig_donut)

        # inv_costs = data['inv_costs']
        # elec_costs_grid = data['elec_costs_grid']
        # elec_costs_ppa_pv = data['elec_costs_ppa_pv']
        # elec_costs_ppa_wind = data['elec_costs_ppa_wind']
        # elec_revenue = data['elec_revenue']
        # peak_power_costs = data['peak_power_costs']

        # fig_donut_costs = go.Figure(data=[go.Pie(labels=['Investment', 'Grid', 'PPA', 'PPA PV', 'PPA Wind', 'Peak Power'],
        #                                 values=[inv_costs, elec_costs_grid, elec_costs_ppa_pv, elec_costs_ppa_wind, elec_revenue, peak_power_costs],
        #                                 hole=0.3,
        #                                 marker_colors=['#00549F', '#407FB7', '#8EBAE5', '#D4E6F1', '#E7EFF6', '#F0F7FC'],
        #                                 textinfo='label+percent',
        #                                 title='')])
        # fig_donut_costs.update_layout(
        #     title=title_str,
        #     title_x=0.5,
        #     title_font=dict(
        #     size=18,
        #     color="black"
        #     )
        # )
        # figs_donut_costs.append(fig_donut_costs)

    return figs_donut_imports, figs_donut_autarky, figs_donut_ppa_usage

