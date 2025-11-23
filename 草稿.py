import pandas as pd
import numpy as np
import statsmodels.api as sm

# ==========================================
# 配置：你的 Excel 文件名
# ==========================================
DATA_FILE = 'Model_Data_Input.xlsx'


class SoybeanRegressionModel:
    def __init__(self, country_name, y_col, cap_col, tariff_col):
        self.country = country_name
        self.y_col = y_col
        self.cap_col = cap_col
        self.tariff_col = tariff_col
        self.model = None
        self.results = None

    def fit(self, data):
        # 准备训练数据 (剔除 2025 年或 Y 为空的数据)
        train_data = data.dropna(subset=[self.y_col])

        if len(train_data) < 3:
            print(f"⚠️ {self.country} 训练数据不足，跳过训练。")
            return None

        Y = train_data[self.y_col]

        # 构造自变量 X: Const, Tariff, Tariff^2, Demand, Capacity
        # 注意：这里会自动读取对应的 Tariff 列 (US/BR/AR)
        X = train_data[[self.tariff_col, 'Tariff_Sq', 'D_china', self.cap_col]]

        # --- 修复：强制添加常数项 (has_constant='add') ---
        # 这样即使巴西关税全是 3.0，它也会强制加一列 const，保证形状匹配
        X = sm.add_constant(X, has_constant='add')
        self.model = sm.OLS(Y, X)
        self.results = self.model.fit()
        return self.results

    def predict(self, future_row):
        # 提取 2025 年的自变量
        X_new = future_row[[self.tariff_col, 'Tariff_Sq', 'D_china', self.cap_col]].to_frame().T
        X_new = sm.add_constant(X_new, has_constant='add')

        # Statsmodels 的 add_constant 对单行数据有时会报错，强制补全 const
        if 'const' not in X_new.columns:
            X_new.insert(0, 'const', 1.0)

        # --- 修正点：使用 .iloc[0] 按位置取值，避免 KeyError ---
        return self.results.predict(X_new).iloc[0]


def main():
    # 1. 读取数据
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"✅ 成功读取数据: {len(df)} 行")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {DATA_FILE}")
        return

    # 预处理
    df['Tariff_Sq'] = df['Tariff_US'] ** 2
    future_df = df[df['Year'] == 2025].copy()

    if future_df.empty:
        print("❌ 错误: 没有2025年数据")
        return

    print(f"\n🔍 检查 2025 输入数据:")
    print(future_df[['Tariff_US', 'D_china', 'Cap_US', 'Cap_BR']].to_string(index=False))
    if future_df.iloc[0]['Tariff_US'] < 10:
        print("⚠️ 警告: 2025年 Tariff_US 只有 {:.1f}，这可能是导致预测为 0 的原因！建议改为 28.0".format(
            future_df.iloc[0]['Tariff_US']))

    print("\n=== 开始模型诊断 ===")

    models = [
        SoybeanRegressionModel('USA', 'EX_US', 'Cap_US', 'Tariff_US'),
        SoybeanRegressionModel('Brazil', 'EX_BR', 'Cap_BR', 'Tariff_US'),  # 这里的 Tariff_US 很关键
        SoybeanRegressionModel('Argentina', 'EX_AR', 'Cap_AR', 'Tariff_US')
    ]

    predictions = {}

    for model in models:
        result = model.fit(df)
        if result:
            # 获取原始预测值 (不加 max 0)
            raw_pred = model.predict(future_df.iloc[0])
            predictions[model.country] = max(0, raw_pred)

            print(f"\n>> {model.country} 模型详情:")
            print(f"   R2: {result.rsquared:.3f}")
            print(f"   原始预测值: {raw_pred:.2f}")  # 这里能看到负数
            print("   回归系数:")
            print(result.params)  # 这里能看到它是怎么算的

    # 输出最终表
    print("\n" + "=" * 30)
    print("     2025年 预测结果")
    print("=" * 30)
    p_world = future_df.iloc[0]['P_world']
    if pd.isna(p_world): p_world = 0

    print(f"{'国家':<10} | {'出口量':<15} | {'出口额':<15}")
    for c, v in predictions.items():
        print(f"{c:<10} | {v:<15.2f} | {v * p_world / 10000:<15.2f}")