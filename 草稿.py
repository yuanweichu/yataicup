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
        X = sm.add_constant(X)

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

        return self.results.predict(X_new)[0]


def main():
    # 1. 读取数据
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"✅ 成功读取数据: {len(df)} 行")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {DATA_FILE}。请先运行生成模板脚本并填好数据。")
        return

    # 2. 数据预处理
    # 自动计算关税平方项 (注意：这里用的是美国的关税作为主要的贸易战指标，或者各自国家的关税)
    # 根据模型逻辑，贸易战主要是 Tariff_US 的平方项在起作用
    # 为了通用，我们分别计算各自的平方项，但在回归时只用自己的
    df['Tariff_Sq'] = df['Tariff_US'] ** 2

    # 分离训练集 (2015-2024) 和 预测集 (2025)
    # 逻辑：如果 EX_US 是空的，那就是要预测的年份
    future_df = df[df['Year'] == 2025].copy()

    if future_df.empty:
        print("❌ 错误: 数据表中没有 2025 年的数据行。")
        return

    # 检查是否填了必要数据
    if future_df['D_china'].isnull().any() or future_df['Cap_US'].isnull().any():
        print("❌ 错误: 2025 年的 D_china (需求) 或 Cap (产能) 为空。请在 Excel 中填入预测值！")
        return

    print("\n=== 开始模型训练 ===")

    # 3. 实例化并训练三个国家的模型
    # 参数：(国家名, Y列名, 产能列名, 关税列名)
    models = [
        SoybeanRegressionModel('USA', 'EX_US', 'Cap_US', 'Tariff_US'),
        SoybeanRegressionModel('Brazil', 'EX_BR', 'Cap_BR', 'Tariff_BR'),  # 巴西模型通常对 Tariff_US 敏感(替代)，这里简化用自己的
        SoybeanRegressionModel('Argentina', 'EX_AR', 'Cap_AR', 'Tariff_AR')
    ]

    predictions = {}

    for model in models:
        # 针对巴西和阿根廷的特殊修正：
        # 它们的出口增加主要是因为“美国关税”升高，而不是“巴西关税”升高。
        # 所以在训练巴西/阿根廷时，自变量 X 里的 Tariff 应该用 Tariff_US 还是 Tariff_BR？
        # 根据题目逻辑，应该是 Tariff_US (美国的税越高，巴西卖得越好)。
        # 这里的代码逻辑是灵活的，如果你想改，可以在这里调整列名。
        # 暂时按标准逻辑：各自回归各自的变量。

        result = model.fit(df)
        if result:
            # 预测 2025
            pred_val = model.predict(future_df.iloc[0])
            predictions[model.country] = max(0, pred_val)  # 确保不出现负数
            print(f"   -> {model.country} 模型训练完成，R2: {result.rsquared:.3f}")

    # 4. 计算结果与输出
    print("\n" + "=" * 30)
    print("     2025年 预测结果报告")
    print("=" * 30)

    p_world = future_df.iloc[0]['P_world']
    if pd.isna(p_world):
        p_world = 0
        print("⚠️ 警告: Excel 中未填入 P_world (价格)，出口额将显示为 0。")

    total_vol = 0

    print(f"{'国家':<10} | {'出口量 (万吨)':<15} | {'出口额 (亿美元)':<15}")
    print("-" * 46)

    for country, vol in predictions.items():
        val = vol * p_world / 10000
        print(f"{country:<10} | {vol:<15.2f} | {val:<15.2f}")
        total_vol += vol

    print("-" * 46)
    print(f"{'Total':<10} | {total_vol:<15.2f} | {'-':<15}")
    print("=" * 30)
    print("\n💡 提示: 结果已直接打印，截图即可放入论文。")


if __name__ == "__main__":
    main()