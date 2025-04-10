import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl

def plot_demand_model(data_model):
    x_values = np.linspace(0, 100, 500)
    data = []
    cost_points = []

    # Generate the demand curves and cost points
    for i in data_model.N:
        y_values = data_model.calculate_demand(x_values, data_model.a_i[i])
        for x, y in  zip(x_values, y_values):
            data.append({"x": x, "y": y, "i": i})
            
        cost = data_model.c_i[i]
        demand_at_cost = data_model.calculate_demand(cost, data_model.a_i[i])
        cost_points.append({"i": i, "cost": cost, "demand": demand_at_cost})
        
    df = pd.DataFrame(data)
    df_cost = pd.DataFrame(cost_points)

    plt.figure(figsize=(10, 6))
    
    # Use a specific palette and get the colors for each item
    colors = sns.color_palette("viridis", len(data_model.N))
    
    # Plot the demand curves (one per item) with corresponding colors.
    sns.lineplot(data=df, x="x", y="y", hue="i", palette=colors)
    
    # Plot the scatter points for observed demand (from d_ik)
    df_points = pd.DataFrame([
         {"i": i, "k": k, "demand": data_model.d_ik[i][k], "price": data_model.r_i[i] * data_model.q_k[k]}
         for i in data_model.N
         for k in data_model.K
    ])
    sns.scatterplot(data=df_points, x="price", y="demand", hue="i",
                    palette=colors, edgecolor="black", s=80, marker="o", legend=False)
    
    # Plot the cost points (using "X" markers)
    sns.scatterplot(data=df_cost, x="cost", y="demand", hue="i",
                    palette=colors, edgecolor="black", s=120, marker="X", legend=False)
    
       
    plt.xlabel("Price per unit", fontsize=14)
    plt.ylabel("Demand (units sold)", fontsize=14)
    #plt.title("Demand Curve for Different Items", fontsize=16)
    
    # Modify legend labels for the line plot
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [f"Item {label}" for label in labels]
    
    plt.legend(handles, labels, title="Legend", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def plot_demand_function(data_model):
   df = pd.DataFrame()

   n = 1000 # Number of data points
   df['Price'] = np.linspace(0, 100, n)

   a_m = [0.1, 0.3, 0.5]
   df['a'] = np.tile(a_m, int(np.ceil(n / len(a_m))))[:n]

   df['Demand'] = data_model.calculate_demand(df['Price'], df['a'])

   sns.lineplot(data=df, x='Price', y='Demand', hue='a', palette='viridis')
   
   plt.xlabel(r"Price ($p$)", fontsize=14)
   plt.ylabel(r"Demand ($d$)", fontsize=14)
   plt.legend(title=r"Price sensitivity parameter ($a$)", fontsize=12)
   plt.show()

def plot_profit_comparison(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    sns.lineplot(x='change_description', y='profit_model', data=df, marker='o', label='Profit Model', linestyle='-', linewidth=2)
    sns.lineplot(x='change_description', y='profit_regular', data=df, marker='s', label='Profit Regular', linestyle='-', linewidth=2)

    # Customize the plot
    plt.xlabel("Change Description")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_profit_margin_comparison(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    sns.lineplot(x='change_description', y='profit_margin_model', data=df, marker='o', label='Profit Margin Model', linestyle='-', linewidth=2)
    sns.lineplot(x='change_description', y='profit_margin_regular', data=df, marker='s', label='Profit Margin Regular', linestyle='-', linewidth=2)

    # Customize the plot
    plt.xlabel("Change Description")
    plt.ylabel("Profit Margin")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_trade_spend_efficiency(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(10, 6))

    sns.lineplot(x='change_description', y='trade_spend_eff', data=df, marker='o', label='Trade Spend Efficiency', linestyle='-', linewidth=2)

    # Customize the plot
    plt.xlabel("Change Description")
    plt.ylabel("Trade Spend Efficiency")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

# def plot_sensitivity_results(csv_file):
#     files = ["a_i.csv", "c_i.csv", "r_i.csv", "trade_spending.csv"]

#     for i in range(len(files)):
#         file = files[i]
#         plot_profit_comparison(file)
#         plot_profit_margin_comparison(file)
#         plot_trade_spend_efficiency(file)

def plot_sensitivity_results(files):
    for i in range(len(files)):
        file = files[i]
        df = pd.read_csv(file)

        # Create a figure with 3 subplots (1 row, 3 columns)
        fig, axes = plt.subplots(nrows=2,ncols= 2, figsize=(14, 10))


        # Plot Profit Comparison
        sns.lineplot(ax=axes[0,0], x='change_description', y='relative_profit_model', data=df, marker='o', label='Model strategy', linestyle='-', linewidth=2)
        sns.lineplot(ax=axes[0,0], x='change_description', y='relative_profit_regular', data=df, marker='s', label='Fixed profit margin strategy', linestyle='-', linewidth=2)
        axes[0,0].set_xlabel("Parameter modification")
        axes[0,0].set_ylabel("Relative change in profit (%)")
        #axes[0,0].set_title("Profit")
        axes[0,0].legend()
        axes[0,0].grid(True)

        # Plot Profit Margin Comparison
        sns.lineplot(ax=axes[0,1], x='change_description', y='relative_profit_margin_model', data=df, marker='o', label='Model strategy', linestyle='-', linewidth=2)
        sns.lineplot(ax=axes[0,1], x='change_description', y='relative_profit_margin_regular', data=df, marker='s', label='Fixed profit margin strategy', linestyle='-', linewidth=2)
        axes[0,1].set_xlabel("Parameter modification")
        axes[0,1].set_ylabel("Relative change in profit Margin (%)")
        #axes[0,1].set_title("Profit Margin")
        axes[0,1].legend()
        axes[0,1].grid(True)

        # Plot Trade Spend Efficiency
        sns.lineplot(ax=axes[1,0], x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label='Model strategy', linestyle='-', linewidth=2)
        sns.lineplot(ax=axes[1,0], x='change_description', y='relative_trade_spend_eff_regular', data=df, marker='s', label='Fixed profit margin strategy', linestyle='-', linewidth=2)
        axes[1,0].set_xlabel("Parameter modification")
        axes[1,0].set_ylabel("Relative change in trade Spend Efficiency (%)")
        #axes[1,0].set_title("Trade Spend Efficiency")
        axes[1,0].legend()
        axes[1,0].grid(True)

        sns.lineplot(ax=axes[1,1], x='change_description', y='change', data=df, marker='^', linestyle='-', linewidth=2)
        axes[1,1].set_xlabel("Parameter modification")
        axes[1,1].set_ylabel("Change in profit (%)")
        #axes[1,1].set_title("Change in profit")
        axes[1,1].grid(True)
        axes[1,1].set_ylim(0,1)
        
        # Adjust layout and show plot
        if file == 'a_i_.csv':
            #fig.suptitle(r'Sensitivity Analysis of $a_i$', fontsize=16)#, fontweight='bold')  # Main title
            plt.savefig('img_results_relative/a_i.png')
        elif file == 'r_i.csv':
            #fig.suptitle(r'Sensitivity Analysis of $r_i$', fontsize=16)#, fontweight='bold')  # Main title
            plt.savefig('img_results_relative/r_i.png')
        elif file == 'c_i.csv':
            plt.savefig('img_results_relative/c_i.png')
            #fig.suptitle(r'Sensitivity Analysis of $c_i$', fontsize=16)#, fontweight='bold')  # Main title
        elif file ==  'trade_spending.csv':
            plt.savefig('img_results_relative/trade_spending.png')
            #fig.suptitle(r'Sensitivity Analysis of $\lambda_i$ and $\Omega_{ivw}$', fontsize=16)#, fontweight='bold')  # Main title
        elif file ==  'eta_i.csv':
            plt.savefig('img_results_relative/eta_i.png')
            #fig.suptitle(r'Sensitivity Analysis of $\eta_i$', fontsize=16)#, fontweight='bold')  # Main title
        elif file ==  'bounds.csv':
            plt.savefig('img_results_relative/bounds.png')
            #fig.suptitle(r'Sensitivity Analysis of $L_{ivw}$ and $U_{ivw}$', fontsize=16)#, fontweight='bold')  # Main title
        elif file ==  'm.csv':
            plt.savefig('img_results_relative/m.png')
            #fig.suptitle(r'Sensitivity Analysis of $m$', fontsize=16)#, fontweight='bold')  # Main title

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap
        plt.show()

def plot_summary_profit():
    #plt.figure(figsize=(13, 6))  # Set figure size
    df = pd.read_csv('a_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$a_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('r_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$r_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('c_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$c_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('eta_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$\eta_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('trade_spending.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$\lambda_i$ & $\Omega_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('bounds.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$L_{ivw}$ & $U_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('m.csv')
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', label=r'$m$', linestyle='-', linewidth=2, alpha=0.5)
    
    plt.xlabel("Parameter modification")
    plt.ylabel("Profit relative to baseline (%)")
    #axes[1,0].set_title("Trade Spend Efficiency")
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.legend()
    plt.grid(True)
    plt.savefig('img_results/summary_profit.png')

def plot_summary_profit_margin():
   # plt.figure(figsize=(13, 6))  # Set figure size
    df = pd.read_csv('a_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$a_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('r_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$r_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('c_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$c_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('eta_i.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$\eta_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('trade_spending.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$\lambda_i$ & $\Omega_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('bounds.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$L_{ivw}$ & $U_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('m.csv')
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', label=r'$m$', linestyle='-', linewidth=2, alpha=0.5)
    
    plt.xlabel("Parameter modification")
    plt.ylabel("Profit margin relative to baseline (%)")
    #axes[1,0].set_title("Trade Spend Efficiency")
    plt.legend()
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.grid(True)
    plt.savefig('img_results/summary_profit_margin.png')


def plot_summary_trade_spend_eff():
    #plt.figure(figsize=(13, 6))  # Set figure size
    df = pd.read_csv('a_i.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$a_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('r_i.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$r_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('c_i.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$c_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('eta_i.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$\eta_i$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('trade_spending.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$\lambda_i$ & $\Omega_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('bounds.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$L_{ivw}$ & $U_{ivw}$', linestyle='-', linewidth=2, alpha=0.5)
    df = pd.read_csv('m.csv')
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', label=r'$m$', linestyle='-', linewidth=2, alpha=0.5)
    
    plt.xlabel("Parameter modification")
    plt.ylabel("Trade spend efficiency relative to baseline (%)")
    plt.legend()
    #axes[1,0].set_title("Trade Spend Efficiency")
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.grid(True)
    plt.savefig('img_results/summary_trade_spend_eff.png')
    

def plot_all_metrics(input_csv, output_png):
    df = pd.read_csv(input_csv)
    
    sns.set_theme(style="whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.tight_layout(pad=6.0)

    # First subplot: relative_trade_spend_eff_model (Profit)
    sns.lineplot(x='change_description', y='relative_profit_model', data=df, marker='o', ax=axs[0, 0])
    axs[0, 0].set_title("Profit relative to baseline")
    axs[0, 0].set_xlabel("Parameter modification")
    axs[0, 0].set_ylabel("Profit (%)")
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].text(0.1, 0.98, '(a)', transform=axs[0, 0].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

    # Second subplot: relative_trade_spend_eff_model (Efficiency)
    sns.lineplot(x='change_description', y='relative_trade_spend_eff_model', data=df, marker='o', ax=axs[0, 1])
    axs[0, 1].set_title("Trade spend efficiency relative to baseline")
    axs[0, 1].set_xlabel("Parameter modification")
    axs[0, 1].set_ylabel("Efficiency (%)")
    axs[0, 1].tick_params(axis='x', rotation=45)
    axs[0, 1].text(0.1, 0.98, '(b)', transform=axs[0, 1].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

    # Third subplot: relative_profit_margin_model
    sns.lineplot(x='change_description', y='relative_profit_margin_model', data=df, marker='o', ax=axs[1, 0])
    axs[1, 0].set_title("Profit margin relative to baseline")
    axs[1, 0].set_xlabel("Parameter modification")
    axs[1, 0].set_ylabel("Profit margin (%)")
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].text(0.1, 0.98, '(c)', transform=axs[1, 0].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')


    # Fourth subplot: change_percent
    df['change_percent'] = df['change'] * 100
    sns.lineplot(x='change_description', y='change_percent', data=df, marker='o', ax=axs[1, 1])
    axs[1, 1].set_title("Relative change in profit ($P_{change}$)")
    axs[1, 1].set_xlabel("Parameter modification")
    axs[1, 1].set_ylabel("Change (%)")
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].text(0.1, 0.98, '(d)', transform=axs[1, 1].transAxes, fontsize=14, verticalalignment='top', fontweight='bold')

    # Save the combined figure
    plt.savefig('img/' + output_png, bbox_inches='tight')
    plt.close()