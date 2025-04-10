import random

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


from Datamodels import DataModel

from Solver import sensitivity_analysis
from Solver import calculate_relative_metrics

from Plot import plot_sensitivity_results
from Plot import plot_demand_model
from Plot import plot_summary_profit
from Plot import plot_summary_profit_margin
from Plot import plot_summary_trade_spend_eff
from Plot import plot_all_metrics

random.seed(42)

# UNCOMMENT TO GENERATE BASELINE DATA
# data_model_baseline = DataModel()
# data_model_baseline.gen_DataModel()
# # plot_demand_model(data_model_baseline) 
# outfile = open('data_model_baseline.pkl', 'wb')
# pickle.dump(data_model_baseline, outfile)
# outfile.close()

# UNCOMMENT TO RUN SENSITIVITY ANALYSIS
# infile = open('data_model_baseline.pkl', 'rb')
# data_model_baseline_pkl = pickle.load(infile)
# sensitivity_analysis(data_model_baseline_pkl, is_bounds=True)

files = ['a_i.csv', 'r_i.csv', 'c_i.csv', 'trade_spending.csv', 'eta_i.csv', 'bounds.csv', 'm.csv']

# UNCOMMENT TO COMPILE RESULTS
#calculate_relative_metrics(files)

outputs = ['a_i.png', 'r_i.png', 'c_i.png', 'trade_spending.png', 'eta_i.png', 'bounds.png', 'm.png']
for file in files:
    df = pd.read_csv(file)
    if 'relative_profit_model' in df.columns:
        df['relative_profit_model_diff'] = df['relative_profit_model'].diff()
        avg_increase = df['relative_profit_model_diff'].mean()
        print(f"Average increase in {file}: {avg_increase}")

plot_all_metrics('a_i.csv', 'a_i.png')

for i in range(len(files)):
    plot_all_metrics(files[i], outputs[i])





