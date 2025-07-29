import pandas as pd

# Your summary table
rank_table = pd.DataFrame({
    'Rank': ['S', 'A', 'B', 'C', 'D'],
    'Avg_Q': [0.633, 0.4517, 0.3216, 0.1489, 0.0373],
    'Avg_Return': [11.42, 7.80, 5.83, 3.49, 2.59],
    'Sharpe': [2.18, 2.11, 1.55, 1.04, 0.72]
})

def classify_signal(q_conf):
    closest_idx = (rank_table['Avg_Q'] - q_conf).abs().idxmin()
    return rank_table.iloc[closest_idx]


new_q = 0.52
result = classify_signal(new_q)
print(result)
