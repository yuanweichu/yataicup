import numpy as np
from scipy.optimize import minimize


class ChipModel:
    def __init__(self):
        self.types = ['High', 'Mid', 'Low']
        self.p = np.array([1200.0, 80.0, 3.0])
        self.c = self.p * 0.6


        self.m0 = np.array([0.5, 8.0, 50.0])

        self.x0 = np.array([0.1, 3.0, 5.0])

        self.D_export = np.array([0.4, 5.0, 10.0])


        self.D_domestic = np.maximum(self.x0 + self.m0 - self.D_export, 0.1)


        self.alpha = np.array([0.0003, 0.0007, 0.0015])
        self.beta = np.array([0.03, 0.10, 0.20])


        self.t_max = 0.50
        self.s_max = 500.0
        self.C_max = self.x0 * 10.0  # 放宽物理上限，允许大幅扩产


        self.f1_min, self.f1_max = 0, 1
        self.f2_min, self.f2_max = 0, 1

    def calculate_vars(self, x_vars):
        t = np.array(x_vars[0:3])
        s = np.array(x_vars[3:6])
        e = x_vars[6]


        d = self.D_domestic + self.D_export
        d[0] = self.D_domestic[0] + self.D_export[0] * (1 - e)


        d_base = self.D_domestic + self.D_export
        ratio = d / d_base




        x_cap = self.x0 + self.beta * s
        x_cap = np.minimum(x_cap, self.C_max)


        x = np.minimum(d, x_cap)


        m = d - x


        m_limit = (self.m0 * ratio) * (1 - self.alpha * (t * 100))
        m_limit = np.maximum(m_limit, 0)

        return t, s, e, m, x, d, m_limit

    def objective_economic(self, x_vars):
        t, s, e, m, x, d, m_limit = self.calculate_vars(x_vars)
        profit = np.sum((self.p - self.c) * x)
        tariff_revenue = np.sum(t * m * self.p)
        subsidy_cost = np.sum(s)
        return profit + tariff_revenue - subsidy_cost

    def objective_security(self, x_vars):
        t, s, e, m, x, d, m_limit = self.calculate_vars(x_vars)


        total_prod_val = np.sum(x * self.p)
        total_supp_val = np.sum(d * self.p)

        if total_supp_val == 0: return 0
        return total_prod_val / total_supp_val

    def combined_objective(self, x_vars, w1, w2):
        f1 = self.objective_economic(x_vars)
        f2 = self.objective_security(x_vars)

        n_f1 = (f1 - self.f1_min) / (self.f1_max - self.f1_min + 1e-6)
        n_f2 = (f2 - self.f2_min) / (self.f2_max - self.f2_min + 1e-6)
        return -(w1 * n_f1 + w2 * n_f2)

    def constraint_import_limit(self, x_vars):

        t, s, e, m, x, d, m_limit = self.calculate_vars(x_vars)
        return m_limit - m





model = ChipModel()
x_start = [0.1] * 3 + [10] * 3 + [0.2]
bounds = [(0, model.t_max)] * 3 + [(0, model.s_max)] * 3 + [(0, 1.0)]

cons = ({'type': 'ineq', 'fun': model.constraint_import_limit})


print("正在校准模型...")
res_f1 = minimize(lambda x: -model.objective_economic(x), x_start, bounds=bounds, constraints=cons, method='SLSQP')
model.f1_max = -res_f1.fun
model.f1_min = model.f1_max * 0.5
res_f2 = minimize(lambda x: -model.objective_security(x), x_start, bounds=bounds, constraints=cons, method='SLSQP')
model.f2_max = -res_f2.fun
model.f2_min = 0


def solve_scenario_forced(w1, w2, scenario_name, fixed_e=None):
    print(f"\n=== 分析情景: {scenario_name} ===")
    current_bounds = list(bounds)
    if fixed_e is not None:
        current_bounds[6] = (fixed_e, fixed_e)

    res = minimize(model.combined_objective, x_start, args=(w1, w2),
                   bounds=current_bounds, constraints=cons, method='SLSQP')

    if res.success:
        t, s, e, m, x, d, m_lim = model.calculate_vars(res.x)
        f1 = model.objective_economic(res.x)
        f2 = model.objective_security(res.x)


        h_rate = x[0] / d[0] if d[0] > 0 else 0

        print(f"  [高端] 关税: {t[0] * 100:.0f}% | 补贴: {s[0]:.2f}亿 | 产量: {x[0]:.3f} | 自给率: {h_rate * 100:.1f}%")
        print(f"  [中端] 关税: {t[1] * 100:.0f}% | 补贴: {s[1]:.2f}亿")
        print(f"  [低端] 关税: {t[2] * 100:.0f}% | 补贴: {s[2]:.2f}亿")
        print(f"  [指标] 经济收益: ${f1:.2f}亿 | 价值自给率: {f2 * 100:.1f}%")
        print(f"  [分析] 进口限制(m_limit): {m_lim[0]:.2f} vs 实际进口(m): {m[0]:.2f}")
    else:
        print("优化失败:", res.message)


# 运行对比
solve_scenario_forced(0.9, 0.1, "Trump (经济优先)", fixed_e=0.0)
solve_scenario_forced(0.1, 0.9, "Biden (科技战)", fixed_e=0.6)