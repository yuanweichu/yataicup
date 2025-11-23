import pandas as pd
import numpy as np
import statsmodels.api as sm

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
        train_data = data.dropna(subset=[self.y_col])

        if len(train_data) < 3:
            print(f"⚠️ {self.country} 训练数据不足，跳过训练。")
            return None

        Y = train_data[self.y_col]

        X = train_data[[self.tariff_col, 'Tariff_Sq', 'D_china', self.cap_col]]


        X = sm.add_constant(X, has_constant='add')
        self.model = sm.OLS(Y, X)
        self.results = self.model.fit()
        return self.results

    def predict(self, future_row):
        X_new = future_row[[self.tariff_col, 'Tariff_Sq', 'D_china', self.cap_col]].to_frame().T
        X_new = sm.add_constant(X_new, has_constant='add')


        if 'const' not in X_new.columns:
            X_new.insert(0, 'const', 1.0)

        return self.results.predict(X_new).iloc[0]


def main():
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"✅ 成功读取数据: {len(df)} 行")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {DATA_FILE}。请先运行生成模板脚本并填好数据。")
        return


    df['Tariff_Sq'] = df['Tariff_US'] ** 2


    future_df = df[df['Year'] == 2025].copy()

    if future_df.empty:
        print("❌ 错误: 数据表中没有 2025 年的数据行。")
        return

    # 检查是否填了必要数据
    if future_df['D_china'].isnull().any() or future_df['Cap_US'].isnull().any():
        print("❌ 错误: 2025 年的 D_china (需求) 或 Cap (产能) 为空。请在 Excel 中填入预测值！")
        return

    print("\n=== 开始模型训练 ===")


    models = [

        SoybeanRegressionModel('USA', 'EX_US', 'Cap_US', 'Tariff_US'),


        SoybeanRegressionModel('Brazil', 'EX_BR', 'Cap_BR', 'Tariff_US'),


        SoybeanRegressionModel('Argentina', 'EX_AR', 'Cap_AR', 'Tariff_US')
    ]
    predictions = {}

    for model in models:

        result = model.fit(df)
        if result:
            pred_val = model.predict(future_df.iloc[0])
            predictions[model.country] = max(0, pred_val)
            print(f"   -> {model.country} 模型训练完成，R2: {result.rsquared:.3f}")


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



if __name__ == "__main__":
    main()