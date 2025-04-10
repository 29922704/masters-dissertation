from amplpy import AMPL
import numpy as np
import pandas as pd

from Datamodels import DataModel
from Plot import plot_demand_model

class Results:
    def __init__(self):
        self.data = pd.DataFrame(columns=["profit_model", "profit_regular", 
                                          "trade_spend_eff_model", "trade_spend_eff_regular", 
                                          "profit_margin_model", "profit_margin_regular", "change_description",
                                          "Change", "cost_model", "cost_regular", "tsi_model", "tsi_regular",
                                          "sale_income_model", "sale_income_regular"])
        self.modifiers = [-0.2,-0.1,0,0.1,0.2]

def vary_param(param, modifier):    
    if not isinstance(param, list):
        return (1 + modifier) * param   

    return [vary_param(element, modifier) for element in param]

def pass_to_ampl(ampl, data_model):
    ampl.set["N"] = list(data_model.N)
    ampl.set["K"] = list(data_model.K)
    ampl.set["W"] = list(data_model.W)

    ampl.param["c_i"] = {i: data_model.c_i[i] for i in data_model.N}
    ampl.param["r_i"] = {i: data_model.r_i[i] for i in data_model.N}
    ampl.param["q_k"] = {k: data_model.q_k[k] for k in data_model.K}
    ampl.param["d_ik"] = {(i, k): data_model.d_ik[i][k] for i in data_model.N for k in data_model.K}
    ampl.param["lambda_i"] = {i: data_model.lambda_i[i] for i in data_model.N}
    ampl.param["Omega_ivw"] = {(i, v, w): data_model.Omega_ivw[i][v][w] for i in data_model.N for v in range(2) for w in data_model.W}
    ampl.param["eta_i"] = {i: data_model.eta_i[i] for i in data_model.N}
    ampl.param["alpha_hik"] = {(h, i, k): data_model.alpha_hik[h][i][k] for h in data_model.N for i in data_model.N for k in data_model.K}
    ampl.param["L_ivw"] = {(i, v, w): data_model.L_ivw[i][v][w] for i in data_model.N for v in range(2) for w in data_model.W}
    ampl.param["U_ivw"] = {(i, v, w): data_model.U_ivw[i][v][w] for i in data_model.N for v in range(2) for w in data_model.W}

def initialise_ampl(ampl, data_model, is_model_solve=True):
    ampl.set_option('solver', 'bonmin')
    ampl.set_option('bonmin_options', 'max_iter=1000 tol=1e-6')
    if is_model_solve:
        ampl.read('model.mod')
    else:
        ampl.read('model_regular.mod')
    pass_to_ampl(ampl, data_model)

def calculate_profit_margin(ampl, total_cost):
    total_proft = ampl.get_objective("Total_Profit").get().value()
    return (total_proft) / (total_proft + total_cost) if total_proft != 0 else 0
 

def calculate_trade_spending_income(ampl):
    N = ampl.get_set("N").get_values().to_list()
    K = ampl.get_set("K").get_values().to_list()
    W = ampl.get_set("W").get_values().to_list()

    c_i = ampl.get_parameter("c_i").get_values().to_dict()
    lambda_i = ampl.get_parameter("lambda_i").get_values().to_dict()
    omega_ivw = ampl.get_variable("omega_ivw").get_values().to_dict()
    O_i = ampl.get_variable("O_i").get_values().to_dict()
    gamma_ik = ampl.get_variable("gamma_ik").get_values().to_dict()
    d_ik = ampl.get_parameter("d_ik").get_values().to_dict()
    phi_i = ampl.get_variable("phi_i").get_values().to_dict()
    
    trade_spending_income = 0.0

    for i in N:
        buy_back = lambda_i[i] * c_i[i] * (O_i[i] - phi_i[i] - sum(gamma_ik[i, k] * (d_ik[i, k] + phi_i[i]) for k in K))

        scan_back = c_i[i] * sum(
            O_i[i] * omega_ivw[i, 0, w] +
            omega_ivw[i, 1, w] * sum(gamma_ik[i, k] * (d_ik[i, k] + phi_i[i]) for k in K)
            for w in W
        )

        trade_spending_income += buy_back + scan_back

    return trade_spending_income

def calculate_sales_income(ampl):
    N = ampl.get_set("N").get_values().to_list()
    K = ampl.get_set("K").get_values().to_list()
    r_i = ampl.get_parameter("r_i").get_values().to_dict()
    q_k = ampl.get_parameter("q_k").get_values().to_dict()
    gamma_ik = ampl.get_variable("gamma_ik").get_values().to_dict()
    d_ik = ampl.get_parameter("d_ik").get_values().to_dict()
    phi_i = ampl.get_variable("phi_i").get_values().to_dict()
    income = 0.0

    for i in N:
        income += sum((q_k[k] * r_i[i] * gamma_ik[i,k] * (d_ik[i,k] + phi_i[i])) for k in K)

    return income

def calculate_trade_spending_efficiency(tsi, sale_income):
    total_income = tsi + sale_income

    # Calculate trade spending efficiency
    trade_spending_efficiency = tsi / total_income

    return trade_spending_efficiency

def calculate_cost(ampl):
    c_i = ampl.get_parameter("c_i").get_values().to_pandas()
    O_i = ampl.get_variable("O_i").get_values().to_pandas()
    
    # Ensure c_i and O_i have the same indices
    if not c_i.index.equals(O_i.index):
        raise ValueError("Indices of c_i and O_i do not match.")
    
    # Calculate the total cost: sum over i in N of c_i * O_i
    return (c_i["c_i"] * O_i["O_i.val"]).sum()    


def sensitivity_solve(baseline_param, baseline_data_model, baseline_param_add = None, 
                      is_a_i = False, is_r_i = False, is_c_i = False, is_trade_spending = False,
                      is_eta_i = False, is_m = False, is_bounds = False):    
    results = Results()

    for i in range(len(results.modifiers)):
        data_model_prime = baseline_data_model.clone()
        param_prime = vary_param(baseline_param, results.modifiers[i])
        if is_a_i:
            data_model_prime.update_demand(param_prime)
            change_description = fr"{results.modifiers[i] + 1}"
        elif is_r_i:
            data_model_prime.r_i = param_prime
            change_description = fr"{results.modifiers[i] + 1}"
        elif is_c_i: 
            data_model_prime.c_i = param_prime
            change_description = fr"{results.modifiers[i] + 1}"
        elif is_trade_spending:
            param_prime_prime = vary_param(baseline_param_add, results.modifiers[i])
            data_model_prime.Omega_ivw = param_prime
            data_model_prime.lambda_i = param_prime_prime
            change_description = fr"{results.modifiers[i] + 1}" 
        elif is_eta_i:
            data_model_prime.eta_i = param_prime
            change_description = fr"{results.modifiers[i] + 1}"
        elif is_m:
            data_model_prime.update_alpha_hik(m_alt = param_prime)
            change_description = fr"{results.modifiers[i] + 1}"
        elif is_bounds:
            data_model_prime.gen_L_U_ivw(modifier=results.modifiers[i], l = 0, u = 0.5)
            change_description = fr"{results.modifiers[i] + 1}"
        else:
            print("Invalid parameter type.")
            return
        
        ampl = AMPL()
        initialise_ampl(ampl, data_model_prime)
        ampl.solve()
        profit_model = ampl.get_objective('Total_Profit').get().value()
        cost_model = calculate_cost(ampl)
        profit_margin_model = calculate_profit_margin(ampl, cost_model)
        tsi_model = calculate_trade_spending_income(ampl)
        sale_income_model = calculate_sales_income(ampl)
        trade_spend_eff_model = calculate_trade_spending_efficiency(tsi=tsi_model, sale_income=sale_income_model)
        
        ampl.close()

        ampl = AMPL()
        initialise_ampl(ampl, data_model_prime, is_model_solve=False)
        ampl.solve()

        profit_regular = ampl.get_objective('Total_Profit').get().value()
        cost_regular = calculate_cost(ampl)
        profit_margin_regular = calculate_profit_margin(ampl, cost_regular)
        
        tsi_regular = calculate_trade_spending_income(ampl)
        sale_income_regular = calculate_sales_income(ampl)
        trade_spend_eff_regular = calculate_trade_spending_efficiency(tsi=tsi_regular, sale_income=sale_income_regular)


        change = (profit_model - profit_regular) / profit_regular


        ampl.close()

        new_row = pd.DataFrame([{
            "profit_model": profit_model,
            "profit_regular": profit_regular,
            "trade_spend_eff_model": trade_spend_eff_model,
            "profit_margin_model": profit_margin_model,
            "profit_margin_regular": profit_margin_regular,
            "change_description": change_description,
            "trade_spend_eff_regular": trade_spend_eff_regular,
            "change": change,
            "cost_model": cost_model,
            "cost_regular": cost_regular,
            "tsi_model": tsi_model,
            "tsi_regular": tsi_regular,
            "sale_income_model": sale_income_model,
            "sale_income_regular": sale_income_regular
        }])

        results.data = pd.concat([results.data, new_row], ignore_index=True)

    if is_a_i:
        results.data.to_csv("a_i.csv", index=False)
    elif is_r_i:
        results.data.to_csv("r_i.csv", index=False)
    elif is_c_i:
        results.data.to_csv("c_i.csv", index = False)
    elif is_trade_spending:
        results.data.to_csv("trade_spending.csv", index = False)
    elif is_eta_i:
        results.data.to_csv("eta_i.csv", index = False)
    elif is_m:
        results.data.to_csv("m.csv", index = False)
    elif is_bounds:
        results.data.to_csv("bounds.csv", index= False)

def sensitivity_analysis(data_model_baseline, is_a_i=False, is_r_i=False, is_c_i=False, is_trade_spending=False, is_eta_i=False, is_m=False, is_bounds=False):
    
    if is_a_i:
        sensitivity_solve(data_model_baseline.a_i, data_model_baseline, is_a_i=True)
    if is_r_i:
        sensitivity_solve(data_model_baseline.r_i, data_model_baseline, is_r_i=True)
    if is_c_i:
        sensitivity_solve(data_model_baseline.c_i, data_model_baseline, is_c_i=True)
    if is_trade_spending:
        sensitivity_solve(data_model_baseline.Omega_ivw, data_model_baseline, baseline_param_add = data_model_baseline.lambda_i, is_trade_spending=True)
    if is_eta_i:
        sensitivity_solve(data_model_baseline.eta_i, data_model_baseline, is_eta_i=True)
    if is_m:
        sensitivity_solve(data_model_baseline.m, data_model_baseline, is_m = True)
    if is_bounds:
        sensitivity_solve(data_model_baseline.L_ivw, data_model_baseline, data_model_baseline.U_ivw, is_bounds=True)

def calculate_relative_metrics(files):
    for file in files:
        df = pd.read_csv(file)
    
        basline_profit_model = df.loc[df['change_description'] == 1.0, 'profit_model'].values[0]
        basline_profit_regular = df.loc[df['change_description'] == 1.0, 'profit_regular'].values[0]
        df['relative_profit_model'] = ((df['profit_model'] - basline_profit_model) / basline_profit_model)*100
        df['relative_profit_regular'] = ((df['profit_regular'] - basline_profit_regular) / basline_profit_regular)*100

        basline_profit_margin_model = df.loc[df['change_description'] == 1.0, 'profit_margin_model'].values[0]
        basline_profit_margin_regular = df.loc[df['change_description'] == 1.0, 'profit_margin_regular'].values[0]
        df['relative_profit_margin_model'] = ((df['profit_margin_model'] - basline_profit_margin_model) / basline_profit_margin_model)*100
        df['relative_profit_margin_regular'] = ((df['profit_margin_regular'] - basline_profit_margin_regular) / basline_profit_margin_regular)*100

        basline_trade_spend_eff_model = df.loc[df['change_description'] == 1.0, 'trade_spend_eff_model'].values[0]
        basline_trade_spend_eff_regular = df.loc[df['change_description'] == 1.0, 'trade_spend_eff_regular'].values[0]
        df['relative_trade_spend_eff_model'] = ((df['trade_spend_eff_model'] - basline_trade_spend_eff_model) / basline_trade_spend_eff_model)*100
        df['relative_trade_spend_eff_regular'] = ((df['trade_spend_eff_regular'] - basline_trade_spend_eff_regular) / basline_trade_spend_eff_regular)*100

        basline_change = df.loc[df['change_description'] == 1.0, 'change'].values[0]
        df['relative_change'] = (df['change'] - basline_change) / basline_change

        df.to_csv(file, index=False)
    
