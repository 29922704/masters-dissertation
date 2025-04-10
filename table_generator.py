import pandas as pd

files = ['a_i.csv', 'r_i.csv', 'c_i.csv', 'trade_spending.csv', 'eta_i.csv', 'bounds.csv', 'm.csv']
ids = [r"$a_i$", r"$r_i$", r"$c_i$", r"$\lambda_i, \Omega_{ivw}$", r"$\eta_i$", r"$L_{ivw}, U_{ivw}$", r"$m$"]

# Read and concatenate all CSV files into one DataFrame
df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# Display the combined DataFrame
print(df)