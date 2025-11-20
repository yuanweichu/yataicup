import pandas as pd
import numpy as np
import statsmodels.api as sm

# ==========================================
# 1. 数据准备 (已包含阿根廷数据)
# ==========================================
data_source = {
    'Year': [2020, 2021, 2022, 2023, 2024],

    # --- 核心自变量 ---
    'Tariff': [2.44, 3.15, 15.60, 18.20, 20.11],
    'D_china': [4800, 4500, 3800, 3200, 2900],

    # --- 各国产能 ---
    'Cap_US': [11000, 10800, 10500, 10300, 10100],
    'Cap_BR': [12000, 12500, 13000, 13500, 14000],
    'Cap_AR': [5000, 5100, 4900, 4800, 5000],  # 阿根廷产能

    # --- 各国对华出口量 (Y) ---
    'EX_US': [5200, 4900, 4100, 3500, 3200],
    'EX_BR': [6000, 6500, 7500, 8000, 8500],
    'EX_AR': [800, 850, 900, 880, 920]  # 阿根廷出口量
}

df = pd.DataFrame(data_source)
# 构建关税二次项
df['Tariff_Sq'] = df['Tariff'] ** 2


class SoybeanRegressionModel:
    def __init__(self, country_name, y_col, cap_col):
        self.country = country_name
        self.y_col = y_col
        self.cap_col = cap_col
        self.model = None
        self.results = None
        self.coeffs = {}

    def fit(self, data):
        # 准备 Y 和 X
        Y = data[self.y_col]
        X = data[['Tariff', 'Tariff_Sq', 'D_china', self.cap_col]]
        X = sm.add_constant(X)

        # OLS 回归
        self.model = sm.OLS(Y, X)
        self.results = self.model.fit()
        self.coeffs = self.results.params
        return self.results

    def check_constraints(self):
        print(f"\n=== {self.country} 模型检验结果 ===")
        beta_1 = self.coeffs.get('Tariff', 0)
        beta_3 = self.coeffs.get('D_china', 0)

        # 符号检验逻辑
        if self.country == 'USA':
            print(f"Tariff系数 < 0 (抑制)? {'✅' if beta_1 < 0 else '❌'} ({beta_1:.4f})")
        else:
            # 巴西和阿根廷应该是正数 (替代效应)
            print(f"Tariff系数 > 0 (替代)? {'✅' if beta_1 > 0 else '❌'} ({beta_1:.4f})")

        print(f"需求系数 > 0? {'✅' if beta_3 > 0 else '❌'} ({beta_3:.4f})")

    def predict_future(self, future_scenario):
        X_new = future_scenario[['Tariff', 'Tariff_Sq', 'D_china', self.cap_col]]
        X_new = sm.add_constant(X_new, has_constant='add')
        return self.results.predict(X_new)


# ==========================================
# 2. 运行模型 (美、巴、阿 三国并列)
# ==========================================

# --- 美国模型 ---
print("\n>>> 正在训练美国模型...")
us_model = SoybeanRegressionModel('USA', 'EX_US', 'Cap_US')
us_model.fit(df)
us_model.check_constraints()

# --- 巴西模型 ---
print("\n>>> 正在训练巴西模型...")
br_model = SoybeanRegressionModel('Brazil', 'EX_BR', 'Cap_BR')
br_model.fit(df)
br_model.check_constraints()

# --- [新增] 阿根廷模型 ---
print("\n>>> 正在训练阿根廷模型...")
ar_model = SoybeanRegressionModel('Argentina', 'EX_AR', 'Cap_AR')
ar_model.fit(df)
ar_model.check_constraints()

# ==========================================
# 3. 2025年 预测应用 (包含阿根廷)
# ==========================================
print("\n=== 2025-2027 多国出口预测 ===")

# 构造未来情景 (必须包含三国的产能预测)
future_data = pd.DataFrame({
    'Year': [2025, 2026, 2027],
    'Tariff': [25.0, 30.0, 35.0],  # 假设关税
    'D_china': [2800, 2700, 2600],  # 假设需求
    'Cap_US': [10000, 9900, 9800],  # 美国产能预测
    'Cap_BR': [14500, 15000, 15500],  # 巴西产能预测
    'Cap_AR': [5100, 5200, 5300],  # [新增] 阿根廷产能预测
    'P_world': [530, 540, 550]  # 价格
})
future_data['Tariff_Sq'] = future_data['Tariff'] ** 2

# 分别预测
pred_us = us_model.predict_future(future_data)
pred_br = br_model.predict_future(future_data)
pred_ar = ar_model.predict_future(future_data)  # [新增] 阿根廷预测

# 汇总结果表
prediction_df = pd.DataFrame({
    'Year': future_data['Year'],
    'US_Export': pred_us,
    'Brazil_Export': pred_br,
    'Argentina_Export': pred_ar,  # [新增]
    'US_Value($B)': pred_us * future_data['P_world'] / 10000,
    'Arg_Value($B)': pred_ar * future_data['P_world'] / 10000  # [新增]
})

print(prediction_df)