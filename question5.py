import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


history_data = {
    'Year': [2020, 2021, 2022, 2023, 2024],
    'Period': ['History', 'History', 'History', 'Baseline', 'Baseline'],  # 2024作为基准

    'GDP_Growth': [-2.2, 5.8, 1.9, 2.5, 2.8],

    'Mfg_Employment_Level': [12188, 12557, 12921, 12980, 12950],

    'Trade_Deficit': [676, 845, 951, 773, 850],

    'CPI': [1.2, 4.7, 8.0, 4.1, 2.9],

    'Investment_Growth': [-4.0, 5.2, 4.5, 3.8, 3.2]
}

df_history = pd.DataFrame(history_data)



def generate_forecast(base_2024):
    future_years = []


    last_gdp = base_2024['GDP_Growth'].values[0]
    last_emp = base_2024['Mfg_Employment_Level'].values[0]
    last_def = base_2024['Trade_Deficit'].values[0]
    last_cpi = base_2024['CPI'].values[0]
    last_inv = base_2024['Investment_Growth'].values[0]


    future_years.append({
        'Year': 2025, 'Period': 'Short-term',
        'GDP_Growth': last_gdp - 0.8,
        'Mfg_Employment_Level': last_emp + 50,
        'Trade_Deficit': last_def - 80,
        'CPI': last_cpi + 1.8,
        'Investment_Growth': last_inv + 0.5
    })


    future_years.append({
        'Year': 2026, 'Period': 'Short-term',
        'GDP_Growth': last_gdp - 0.3,
        'Mfg_Employment_Level': last_emp + 120,
        'Trade_Deficit': last_def - 120,
        'CPI': last_cpi + 1.2,
        'Investment_Growth': last_inv + 1.0
    })


    future_years.append({
        'Year': 2027, 'Period': 'Mid-term',
        'GDP_Growth': last_gdp + 0.2,
        'Mfg_Employment_Level': last_emp + 250,
        'Trade_Deficit': last_def - 150,
        'CPI': last_cpi + 0.5,
        'Investment_Growth': last_inv + 1.5
    })


    for y in [2028, 2029]:
        future_years.append({
            'Year': y, 'Period': 'Mid-term',
            'GDP_Growth': 3.0,
            'Mfg_Employment_Level': future_years[-1]['Mfg_Employment_Level'] + 150,
            'Trade_Deficit': 650,
            'CPI': 2.5,
            'Investment_Growth': 5.5
        })

    return pd.DataFrame(future_years)



df_future = generate_forecast(df_history[df_history['Year'] == 2024])


df_all = pd.concat([df_history, df_future], ignore_index=True)
df_all.set_index('Year', inplace=True)


df_model = df_all[df_all.index >= 2024].copy()


positive_cols = ['GDP_Growth', 'Mfg_Employment_Level', 'Investment_Growth']
negative_cols = ['Trade_Deficit', 'CPI']
indicators = positive_cols + negative_cols

print("建模数据矩阵 X (包含真实基准 + 预测值):")
print(df_model[indicators])
print("-" * 50)



def calculate_entropy_weights(df, pos_cols, neg_cols):
    X = df.copy()
    normalized_X = pd.DataFrame(index=X.index)


    for col in pos_cols:
        normalized_X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min()) + 1e-5


    for col in neg_cols:
        normalized_X[col] = (X[col].max() - X[col]) / (X[col].max() - X[col].min()) + 1e-5

    P = normalized_X.div(normalized_X.sum(axis=0), axis=1)
    n = len(df)
    k = 1 / np.log(n)
    E = -k * (P * np.log(P)).sum(axis=0)
    d = 1 - E
    weights = d / d.sum()

    return weights, normalized_X



weights, norm_df_entropy = calculate_entropy_weights(df_model[indicators], positive_cols, negative_cols)

print("熵权法计算出的指标权重 w_j:")
print(weights.sort_values(ascending=False))
print("-" * 50)



def calculate_topsis_score(norm_df, weights):
    V = norm_df.mul(weights, axis=1)
    Z_plus = V.max()
    Z_minus = V.min()
    D_plus = np.sqrt(((V - Z_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - Z_minus) ** 2).sum(axis=1))
    C = D_minus / (D_plus + D_minus)
    return C


scores = calculate_topsis_score(norm_df_entropy, weights)
df_model['Score_C'] = scores

print("TOPSIS 综合评分结果:")
print(df_model[['Period', 'Score_C']])
print("-" * 50)




score_baseline = df_model[df_model['Period'] == 'Baseline']['Score_C'].mean()
score_short = df_model[df_model['Period'] == 'Short-term']['Score_C'].mean()
score_mid = df_model[df_model['Period'] == 'Mid-term']['Score_C'].mean()

delta_C_short = score_short - score_baseline
delta_C_mid = score_mid - score_baseline




hist_avg_inv = df_history[df_history['Year'] < 2025]['Investment_Growth'].mean()
g_threshold_inv = max(hist_avg_inv, 2.0)


emp_start = df_history.loc[df_history['Year'] == 2020, 'Mfg_Employment_Level'].values[0]
emp_end = df_history.loc[df_history['Year'] == 2024, 'Mfg_Employment_Level'].values[0]
hist_emp_cagr = ((emp_end / emp_start) ** (1 / 4) - 1) * 100
r_threshold_emp = max(hist_emp_cagr, 1.0)

print(f"【动态阈值设定】")
print(f"基于历史数据 (2020-2024):")
print(f"  - 投资增速阈值 (g_threshold): {g_threshold_inv:.2f}%")
print(f"  - 就业增长阈值 (r_threshold): {r_threshold_emp:.2f}%")
print("-" * 30)


avg_inv_policy = df_model[df_model['Period'] != 'Baseline']['Investment_Growth'].mean()


emp_2029 = df_model.loc[2029, 'Mfg_Employment_Level']
emp_2024 = df_model.loc[2024, 'Mfg_Employment_Level']
future_emp_cagr = ((emp_2029 / emp_2024) ** (1 / 5) - 1) * 100

print("【分析报告】")
print(f"基准评分 (2024): {score_baseline:.4f}")
print(f"短期评分 (2025-26): {score_short:.4f} (ΔC = {delta_C_short:.4f})")
print(f"中期评分 (2027-29): {score_mid:.4f} (ΔC = {delta_C_mid:.4f})")
print("-" * 30)

print("制造业回流判定:")
cond1 = delta_C_mid > 0
cond2 = future_emp_cagr > r_threshold_emp
cond3 = avg_inv_policy > g_threshold_inv

print(f"条件1: 综合经济状态改善 (ΔC_mid > 0)? {'YES' if cond1 else 'NO'}")
print(
    f"条件2: 制造业就业年均增速 ({future_emp_cagr:.2f}%) > 历史阈值 ({r_threshold_emp:.2f}%)? {'YES' if cond2 else 'NO'}")
print(
    f"条件3: 本土投资平均增速 ({avg_inv_policy:.2f}%) > 历史阈值 ({g_threshold_inv:.2f}%)? {'YES' if cond3 else 'NO'}")

if cond1 and cond2 and cond3:
    if delta_C_mid > 0.2:
        effect = "强效应"
    else:
        effect = "中等效应"
    print(f"\n结论: 美国关税政策对制造业回流产生了【{effect}】。")
else:
    print(f"\n结论: 政策【未能有效】推动制造业回流 (或仅产生弱/负效应)。")


plt.figure(figsize=(10, 6))
plt.plot(df_model.index, df_model['Score_C'], marker='o', label='Comprehensive Score')
plt.axhline(y=score_baseline, color='r', linestyle='--', label='Baseline (2024)')
plt.title('Economic Impact Score Prediction (2024-2029)')
plt.grid(True)
plt.legend()
plt.show()