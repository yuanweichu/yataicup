import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings


warnings.filterwarnings("ignore")



dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
n_months = len(dates)

np.random.seed(42)
data = {
    'Date': dates,
    'Electronics_Import': np.linspace(50, 60, n_months) + np.sin(np.linspace(0, 20, n_months)) * 5 + np.random.normal(0,
                                                                                                                      1,
                                                                                                                      n_months),
    'RawMaterials_Import': np.linspace(30, 35, n_months) + np.random.normal(0, 0.5, n_months),
    'Auto_Import': np.linspace(40, 45, n_months) + np.sin(np.linspace(0, 10, n_months)) * 3 + np.random.normal(0, 1,
                                                                                                               n_months)
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

print(">>> 历史数据预览 (2020-2024):")
print(df.head())
print("-" * 50)


# 保持原样
forecast_steps = 60
forecast_dates = pd.date_range(start='2025-01-01', periods=forecast_steps, freq='M')


def run_arima_forecast(series, order=(1, 1, 1)):
    result = adfuller(series.dropna())
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast


baseline_forecasts = pd.DataFrame(index=forecast_dates)

print("正在进行 ARIMA 预测...")
for col in df.columns:
    forecast_values = run_arima_forecast(df[col], order=(1, 1, 0))
    baseline_forecasts[col] = forecast_values

print(">>> 基准进口额预测完成 (M_hat_base)")


# 辅助函数：生成 Logistic S型曲线
def get_smooth_multiplier(steps, transition_center, steepness, start_val, end_val):
    x = np.arange(steps)

    sigmoid = 1 / (1 + np.exp(-steepness * (x - transition_center)))

    return start_val + (end_val - start_val) * sigmoid

params = {
    'Electronics_Import': {'tau_old': 0.025, 'tau_new': 0.20, 'eta': -1.2, 'ER': 0.9},
    'RawMaterials_Import': {'tau_old': 0.025, 'tau_new': 0.15, 'eta': -0.4, 'ER': 0.95},
    'Auto_Import': {'tau_old': 0.025, 'tau_new': 0.25, 'eta': -0.8, 'ER': 0.9}
}

revenue_base_df = pd.DataFrame(index=forecast_dates)
revenue_policy_df = pd.DataFrame(index=forecast_dates)


smooth_multipliers = get_smooth_multiplier(steps=forecast_steps,
                                           transition_center=24,
                                           steepness=0.3,
                                           start_val=0.15,
                                           end_val=1.2)



for col in df.columns:
    p = params[col]
    M_base = baseline_forecasts[col]


    dynamic_eta = p['eta'] * smooth_multipliers


    dynamic_eta_series = pd.Series(data=dynamic_eta, index=forecast_dates)


    rate_change_pct = (p['tau_new'] - p['tau_old']) / p['tau_old']


    volume_change_factor = 1 + dynamic_eta_series * rate_change_pct


    volume_change_factor = volume_change_factor.clip(lower=0.1)

    M_adj = M_base * volume_change_factor


    TR_base = p['tau_old'] * M_base * p['ER']
    TR_policy = p['tau_new'] * M_adj * p['ER']

    revenue_base_df[f'{col}_Revenue'] = TR_base
    revenue_policy_df[f'{col}_Revenue'] = TR_policy


total_revenue_base = revenue_base_df.sum(axis=1)
total_revenue_policy = revenue_policy_df.sum(axis=1)


analysis_df = pd.DataFrame({
    'Base_Revenue': total_revenue_base,
    'Policy_Revenue': total_revenue_policy
})
analysis_df['Net_Change'] = analysis_df['Policy_Revenue'] - analysis_df['Base_Revenue']

short_term = analysis_df['2025':'2026']
medium_term = analysis_df['2027':'2029']

delta_TR_short = short_term['Net_Change'].sum()
delta_TR_medium = medium_term['Net_Change'].sum()
delta_TR_total = delta_TR_short + delta_TR_medium

print("\n" + "=" * 50)
print("PREDICTION RESULTS (预测结果 - 方案A: 动态弹性)")
print("=" * 50)
print(f"1. 短期净变化 (2025-2026): {delta_TR_short:.2f} 亿美元 (预期为正)")
print(f"2. 中期净变化 (2027-2029): {delta_TR_medium:.2f} 亿美元 (预期为负或大幅下降)")
print(f"3. 总净变化 (特朗普任期):   {delta_TR_total:.2f} 亿美元")
print("-" * 50)


plt.figure(figsize=(12, 6))

# 绘制曲线
plt.plot(analysis_df.index, analysis_df['Base_Revenue'], label='Base Scenario (Low Tariff)',
         linestyle='--', color='blue', linewidth=2)
plt.plot(analysis_df.index, analysis_df['Policy_Revenue'], label='Policy Scenario (High Tariff)',
         color='red', linewidth=2)

plt.fill_between(analysis_df.index,
                 analysis_df['Base_Revenue'],
                 analysis_df['Policy_Revenue'],
                 where=(analysis_df['Policy_Revenue'] > analysis_df['Base_Revenue']),
                 color='green', alpha=0.3, label='Revenue Gain (Short Term)')

plt.fill_between(analysis_df.index,
                 analysis_df['Base_Revenue'],
                 analysis_df['Policy_Revenue'],
                 where=(analysis_df['Policy_Revenue'] <= analysis_df['Base_Revenue']),
                 color='red', alpha=0.3, label='Revenue Loss (Medium Term)')

plt.title('Prediction of US Tariff Revenue: Short-Term Gain vs. Medium-Term Pain', fontsize=14)
plt.ylabel('Monthly Tariff Revenue (Billion $)')
plt.xlabel('Year')


plt.axvline(pd.to_datetime('2027-01-01'), color='black', linestyle=':', linewidth=2)
plt.text(pd.to_datetime('2026-01-01'), analysis_df['Policy_Revenue'].max() * 0.9,
         'Short Term\n(Rigid Demand)', ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.text(pd.to_datetime('2028-06-01'), analysis_df['Policy_Revenue'].max() * 0.9,
         'Medium Term\n(Substitution Effect)', ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()