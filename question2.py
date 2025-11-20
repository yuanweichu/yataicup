import numpy as np
import pandas as pd


# ==========================================
# 第一部分：AHP 层次分析法 (确定权重)
# ==========================================

class AHP:
    def __init__(self, criteria_names, judgment_matrix):
        self.criteria = criteria_names
        self.n = len(criteria_names)
        self.matrix = np.array(judgment_matrix)
        self.RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

    def calculate_weights(self):
        # 1. 计算权重 (使用特征值法)
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
        max_index = np.argmax(np.real(eigenvalues))
        max_eigenvalue = np.real(eigenvalues[max_index])
        eigenvector = np.real(eigenvectors[:, max_index])

        # 归一化特征向量得到权重
        self.weights = eigenvector / np.sum(eigenvector)

        # 2. 一致性检验
        CI = (max_eigenvalue - self.n) / (self.n - 1)
        RI = self.RI_dict.get(self.n, 1.41)  # 默认取值
        CR = CI / RI if RI != 0 else 0

        return self.weights, max_eigenvalue, CI, RI, CR

    def print_results(self):
        w, lam, ci, ri, cr = self.calculate_weights()
        print("\n" + "=" * 40)
        print(" Step 1: AHP 权重计算结果")
        print("=" * 40)
        print(f"最大特征值 (lambda_max): {lam:.4f}")
        print(f"一致性指标 (CI): {ci:.4f}")
        print(f"一致性比率 (CR): {cr:.4f}")

        if cr < 0.1:
            print("结果判定: [通过] 一致性检验 (CR < 0.1)")
        else:
            print("结果判定: [不通过] 请调整判断矩阵")

        print("-" * 40)
        print("各项指标权重:")
        for name, weight in zip(self.criteria, w):
            print(f"  - {name}: {weight:.4f}")
        return w


# ==========================================
# 第二部分：TOPSIS 优劣解距离法 (评价方案)
# ==========================================

class TOPSIS:
    def __init__(self, data, weights, criteria_types):
        """
        data: DataFrame, 原始数据矩阵 (行=方案, 列=指标)
        weights: array, AHP计算出的权重
        criteria_types: list, 指标类型 ('min' 表示成本型/越小越好, 'max' 表示效益型/越大越好)
        """
        self.data = data
        self.weights = weights
        self.types = criteria_types
        self.norm_matrix = None
        self.weighted_matrix = None

    def normalize(self):
        # 1. 向量归一化 (对应图片中的公式 rij = xij / sqrt(sum(x^2)))
        mat = self.data.values.astype(float)
        # 计算每一列的平方和开根号
        norm_factor = np.sqrt(np.sum(mat ** 2, axis=0))
        self.norm_matrix = mat / norm_factor
        return self.norm_matrix

    def apply_weights(self):
        # 2. 加权标准化
        self.weighted_matrix = self.norm_matrix * self.weights
        return self.weighted_matrix

    def calculate_distance_and_score(self):
        # 3. 确定理想解 (Z+) 和 负理想解 (Z-)
        z_positive = []
        z_negative = []

        num_criteria = self.weighted_matrix.shape[1]

        for j in range(num_criteria):
            col = self.weighted_matrix[:, j]
            if self.types[j] == 'max':  # 效益型: 越大越好
                z_positive.append(np.max(col))
                z_negative.append(np.min(col))
            else:  # 成本型: 越小越好 ('min')
                z_positive.append(np.min(col))
                z_negative.append(np.max(col))

        z_positive = np.array(z_positive)
        z_negative = np.array(z_negative)

        # 4. 计算欧氏距离 (D+, D-)
        # 对每一行(方案)计算到理想解的距离
        d_positive = np.sqrt(np.sum((self.weighted_matrix - z_positive) ** 2, axis=1))
        d_negative = np.sqrt(np.sum((self.weighted_matrix - z_negative) ** 2, axis=1))

        # 5. 计算贴近度 (C)
        scores = d_negative / (d_positive + d_negative)

        return z_positive, z_negative, d_positive, d_negative, scores

    def print_results(self):
        self.normalize()
        self.apply_weights()
        z_pos, z_neg, d_pos, d_neg, scores = self.calculate_distance_and_score()

        print("\n" + "=" * 40)
        print(" Step 2: TOPSIS 评价计算结果")
        print("=" * 40)

        # 创建结果 DataFrame
        res_df = pd.DataFrame({
            '方案': self.data.index,
            '正理想距离 (D+)': d_pos,
            '负理想距离 (D-)': d_neg,
            '贴近度 (Score)': scores
        })

        # 按分数降序排列
        res_df = res_df.sort_values(by='贴近度 (Score)', ascending=False)

        print("最终排名:")
        print(res_df.to_string(index=False, float_format="%.4f"))

        print("-" * 40)
        best_option = res_df.iloc[0]['方案']
        print(f"结论: 最优出口模式为 [{best_option}]")
        print("=" * 40)


# ==========================================
# 主程序：输入数据并运行
# ==========================================

if __name__ == "__main__":
    # --------------------------------------
    # 1. 设置 AHP 数据 (对应你提供的图片1)
    # --------------------------------------
    criteria = ["价格", "关税成本", "非关税措施效果", "市场份额稳定性"]

    # 判断矩阵 (1-9 标度法)
    # 顺序: 价格, 关税成本, 非关税, 市场份额
    judgment_matrix = [
        [1, 1 / 3, 1 / 5, 1 / 2],  # 价格
        [3, 1, 1 / 2, 2],  # 关税成本
        [5, 2, 1, 3],  # 非关税 (最重要)
        [2, 1 / 2, 1 / 3, 1]  # 市场份额
    ]

    # 运行 AHP
    ahp_model = AHP(criteria, judgment_matrix)
    weights = ahp_model.print_results()

    # --------------------------------------
    # 2. 设置 TOPSIS 数据 (对应你提供的图片6 - 修正后的数据逻辑)
    # --------------------------------------
    # 方案名称
    alternatives = ["直接出口", "墨西哥代工"]

    # 原始数据矩阵 (行对应方案, 列对应指标)
    # 建议在这里填入论文中估算的具体数值
    raw_data = [
        # 价格(万), 关税成本(万), 非关税效果(0-1), 稳定性(0-1)
        [3.5, 0.8, 0.3, 0.6],  # 直接出口
        [3.2, 0.3, 0.8, 0.7]  # 墨西哥代工
    ]

    df_data = pd.DataFrame(raw_data, columns=criteria, index=alternatives)

    print("\n原始决策矩阵:")
    print(df_data)

    # 指标类型定义
    # 价格: min (越低越好)
    # 关税成本: min (越低越好)
    # 非关税效果: max (越高越好)
    # 市场稳定性: max (越高越好)
    types = ['min', 'min', 'max', 'max']

    # 运行 TOPSIS
    topsis_model = TOPSIS(df_data, weights, types)
    topsis_model.print_results()