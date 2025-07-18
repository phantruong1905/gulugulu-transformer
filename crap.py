import pandas as pd
df = pd.read_excel('E:/stock.xlsx')  # Replace with your Excel file path
unique_list = df.iloc[:, 0].unique().tolist()  # First column, unique values as list
print(unique_list)  # Outputs: ['LHG', 'VOS', 'PLC', 'FOX', 'KLB', 'AGG']