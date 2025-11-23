import numpy as np
import pandas as pd


class AHP_TOPSIS_Model:
    def __init__(self):
        self.criteria = ["价格", "关税成本", "非关税措施效果", "市场份额稳定性"]
        self.alternatives = ["直接出口模式", "墨西哥代工模式"]
        self.types = ['min', 'min', 'max', 'max']

    def run_model(self, scenario_name, P_d, T_d, P_m, T_m, Q_d_vol, Q_m_vol):
        W = np.array([0.097, 0.277, 0.483, 0.143])
        raw_data = np.array([
            [P_d, T_d, 0.3, 0.9],
            [P_m, T_m, 0.8, 0.6]
        ])

        denom = np.sqrt(np.sum(raw_data ** 2, axis=0))
        denom[denom == 0] = 1e-9  # 避免除以0
        norm_matrix = raw_data / denom
        weighted_matrix = norm_matrix * W

        Z_plus = []
        Z_minus = []
        for i in range(len(self.criteria)):
            col = weighted_matrix[:, i]
            if self.types[i] == 'max':
                Z_plus.append(np.max(col))
                Z_minus.append(np.min(col))
            else:
                Z_plus.append(np.min(col))
                Z_minus.append(np.max(col))

        D_plus = np.sqrt(np.sum((weighted_matrix - np.array(Z_plus)) ** 2, axis=1))
        D_minus = np.sqrt(np.sum((weighted_matrix - np.array(Z_minus)) ** 2, axis=1))

        # 贴近度计算
        total_dist = D_plus + D_minus
        C = np.divide(D_minus, total_dist, out=np.zeros_like(D_minus), where=total_dist != 0)

        print(f"\n{'=' * 50}")
        print(f">>> 场景分析: {scenario_name}")
        print(f"{'=' * 50}")
        print(f"    [输入] 直接出口: 价格={P_d}, 关税={T_d:.3f}")
        print(f"    [输入] 墨西哥代工: 价格={P_m}, 关税={T_m:.3f}")
        print(f"-" * 30)
        print(f"    直接出口竞争力得分 (C1): {C[0]:.4f}")
        print(f"    墨西哥代工竞争力得分 (C2): {C[1]:.4f}")

        self.calculate_market_impact(C, Q_d_vol, Q_m_vol)

        return C

    def calculate_market_impact(self, C, Q_d, Q_m):
        S_J0 = 0.20
        beta = 0.6
        I_total = Q_d + Q_m



        C_1, C_2 = C[0], C[1]
        weighted_score = (C_1 * Q_d + C_2 * Q_m) / (Q_d + Q_m)


        S_J_new = S_J0 * weighted_score

        print(f"-" * 30)
        print(f"    [模型扩展] 市场份额与产业影响分析")
        print(f"    1. 综合竞争力指数: {weighted_score:.4f}")
        print(f"    2. 日本车在美份额预测 (S_J):")
        print(f"       初始份额: {S_J0:.1%}")
        print(f"       预测份额: {S_J_new:.1%} (下降了 {(S_J0 - S_J_new):.1%})")



        loss_ratio = 1 - (S_J_new / S_J0)
        Delta_S_US = beta * loss_ratio

        print(f"    3. 对美国本土汽车工业的影响 (Delta S_US):")
        print(f"       流失的日本车份额: {loss_ratio:.1%}")
        print(f"       >>> 美国本土车企预计增加份额: +{Delta_S_US:.2%} <<<")
        print(f"       (结论: 关税政策起到了保护本土工业的作用，但程度受弹性系数限制)")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    model = AHP_TOPSIS_Model()


    scenarios = [
        {
            "name": "场景一 ",
            # 日本直接出口: 价格4.1万, 关税20.11% (赛题平均值)
            "P_d": 4.1, "T_d": 4.1 * 0.2011,
            # 墨西哥代工: 价格2.8万, 关税10% (赛题基准关税)
            "P_m": 2.8, "T_m": 2.8 * 0.10,
            # 假设销量权重 (Q): 以前直接出口多，现在可能持平或墨西哥多
            "Q_d": 50, "Q_m": 50
        },
        {
            "name": "场景二 ",
            # 日本直接出口: 价格4.2万
            "P_d": 4.2, "T_d": 4.2 * 0.2011,
            # 墨西哥代工: 价格3.16万
            "P_m": 3.16, "T_m": 3.16 * 0.10,
            "Q_d": 40, "Q_m": 60
        }
    ]

    for s in scenarios:
        model.run_model(s["name"], s["P_d"], s["T_d"], s["P_m"], s["T_m"], s["Q_d"], s["Q_m"])