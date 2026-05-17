# 编程实践题：空间回归分析

本练习使用中国地级市数据，围绕 Week 9 课程内容设计。

## 数据说明

- 城市边界：`data/china/city.shp`（372个地级市面状多边形）
- 属性数据：`data/china/地级市数据.csv`（287个城市 × 19年面板数据）
- 分析年份：2021 年
- 因变量：`经济发展水平`（ln(人均GDP)）

---

## 题目一：OLS 回归基础与残差诊断（基础）

**目标**：建立 OLS 回归模型，检验经典假设是否成立。

### 要求

1. 加载数据并合并 shapefile 和 CSV（筛选 2021 年）
2. 选择因变量 `经济发展水平`，自变量至少选择 3 个（如 `城镇化水平`、`金融发展程度`、`产业结构高级化`）
3. 使用 `statsmodels` 拟合 OLS 回归模型
4. 进行完整的残差诊断：
   - 残差 vs 拟合值图（检验同方差性）
   - 残差 Q-Q 图（检验正态性）
   - 计算 VIF（方差膨胀因子，检验多重共线性）
   - Durbin-Watson 检验（检验残差自相关）
5. 计算并绘制 Moran's I 散点图，检验残差的空间自相关

### 参考代码提示

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

# OLS 回归
X = df[['城镇化水平', '金融发展程度', '产业结构高级化']]
X = sm.add_constant(X)
y = df['经济发展水平']
model = sm.OLS(y, X).fit()
print(model.summary())

# VIF
for i, col in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF({col}) = {vif:.2f}")

# Durbin-Watson
dw = durbin_watson(model.resid)
print(f"Durbin-Watson = {dw:.4f}")
# DW ≈ 2 表示无自相关，< 2 正自相关，> 2 负自相关
```

### 思考题

- 如果残差存在空间自相关，OLS 的估计结果还可靠吗？为什么？
- Durbin-Watson 统计量检验的是什么类型的自相关？它适合空间数据吗？

---

## 题目二：空间权重矩阵构建与空间回归模型（中等）

**目标**：构建空间权重矩阵，拟合 SAR、SEM、SDM 模型并比较。

### 要求

1. 构建空间权重矩阵（Queen 邻接 + KNN(4)），行标准化
2. 使用 `spreg` 或 `spdep` 拟合以下模型：
   - **OLS**（基准模型）
   - **SAR（空间滞后模型）**
   - **SEM（空间误差模型）**
   - **SDM（空间杜宾模型）**
3. 比较各模型的拟合优度（R²、AIC、Log-Likelihood）
4. 使用 LM 检验（LM-Lag、LM-Error、Robust LM-Lag、Robust LM-Error）判断应选择哪种空间模型
5. 绘制各模型残差的 Moran's I 散点图，对比空间自相关是否消除

### 参考代码提示

```python
from libpysal.weights import Queen, KNN
from spreg import OLS, ML_Lag, ML_Error, ML_Error_Lag

# 构建权重矩阵
w = Queen.from_dataframe(gdf)
w.transform = 'r'

# SAR 模型
sar = ML_Lag(y.values, X.values, w, name_y='经济发展水平', name_x=X.columns.tolist())
print(sar.summary)

# SEM 模型
sem = ML_Error(y.values, X.values, w, name_y='经济发展水平', name_x=X.columns.tolist())
print(sem.summary)

# SDM 模型（需要包含空间滞后解释变量 WX）
# ML_Error_Lag 可以同时估计 SAR 和 SEM 的参数
sdx = ML_Error_Lag(y.values, X.values, w, name_y='经济发展水平', name_x=X.columns.tolist())
print(sdx.summary)
```

### 思考题

- 如果 LM-Lag 显著但 LM-Error 不显著，应该选择哪个模型？
- SAR 的 rho 参数为正意味着什么？
- 为什么 SAR 的 R² 通常高于 OLS？

---

## 题目三：空间效应分解（中等）

**目标**：对 SAR/SDM 模型的回归系数进行直接效应和间接效应分解。

### 要求

1. 拟合 SAR 或 SDM 模型
2. 计算直接效应、间接效应（空间溢出效应）和总效应
3. 用表格展示各变量的效应分解结果
4. 绘制各城市间接效应的空间分布图

### 参考代码提示

```python
import numpy as np

# SAR 效应分解
# 简化形式: y = (I - rho*W)^{-1} * (X*beta + epsilon)
# 直接效应: diag((I - rho*W)^{-1}) 的均值 * beta
# 间接效应: (行和 - diag)((I - rho*W)^{-1}) 的均值 * beta

def sar_effects(rho, w, beta, n):
    """计算 SAR 模型的直接/间接/总效应"""
    I = np.eye(n)
    S = np.linalg.inv(I - rho * w.sparse.toarray())
    
    # 直接效应
    direct = np.diag(S).mean() * beta
    
    # 间接效应
    row_sums = S.sum(axis=1)
    indirect = (row_sums - np.diag(S)).mean() * beta
    
    # 总效应
    total = direct + indirect
    
    return direct, indirect, total

# 对每个自变量计算
for i, var in enumerate(X.columns[1:]):  # 跳过常数项
    d, ind, t = sar_effects(rho, w, beta[i], n)
    print(f"{var}: 直接={d:.4f}, 间接={ind:.4f}, 总={t:.4f}")
```

### 思考题

- 为什么 SAR 模型中某个变量的间接效应不为零？
- 直接效应和 OLS 回归系数一样吗？为什么？
- 如果一个变量的间接效应为负，在政策上意味着什么？

---

## 题目四：地理加权回归 GWR（中等）

**目标**：使用 GWR 捕捉空间异质性，比较局部回归系数的空间变化。

### 要求

1. 使用 `mgwr` 包拟合 GWR 模型
2. 选择合适的带宽（使用 AIC 准则）
3. 绘制回归系数（至少选一个自变量）的空间分布图
4. 绘制局部 R² 的空间分布图
5. 比较 GWR 与 OLS 的全局模型，讨论空间异质性的存在

### 参考代码提示

```python
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# 准备数据
coords = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))
X_gwr = df[['城镇化水平', '金融发展程度', '产业结构高级化']].values
y_gwr = df['经济发展水平'].values

# 带宽选择（AIC 准则）
bw = Sel_BW(coords, y_gwr, X_gwr).search()
print(f"最优带宽: {bw:.0f}")

# GWR 模型拟合
gwr_model = GWR(coords, y_gwr, X_gwr, bw).fit()

# 局部 R²
local_r2 = gwr_model.localR2

# 局部系数（第一个自变量的系数）
local_coef_1 = gwr_model.params[:, 1]  # 第 1 列 = 第一个自变量

# 绘制局部系数空间分布
gdf['local_r2'] = local_r2
gdf['coef_城镇化水平'] = local_coef_1

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
gdf.plot(column='local_r2', cmap='RdYlGn', legend=True, ax=axes[0])
axes[0].set_title('局部 R² 空间分布')
gdf.plot(column='coef_城镇化水平', cmap='RdYlGn', legend=True, ax=axes[1])
axes[1].set_title('城镇化水平系数的空间分布')
plt.show()
```

### 思考题

- GWR 的局部 R² 在哪些区域较低？可能的原因是什么？
- 某个自变量的系数在不同区域符号不同（正变负），说明了什么？
- GWR 的局限是什么？它能解决空间依赖性问题吗？

---

## 题目五（进阶）：模型诊断与选择的完整流程（进阶）

**目标**：实现从 OLS 诊断到空间模型选择、再到效应分解的完整分析流程。

### 要求

1. 拟合 OLS 模型，进行残差诊断（同方差、正态性、多重共线性）
2. 检验残差的空间自相关（Moran's I）
3. 如果存在空间自相关，使用 LM 检验确定模型类型
4. 拟合对应的空间回归模型（SAR/SEM/SDM）
5. 对比 OLS 和空间模型的 AIC/BIC
6. 对 SAR/SDM 进行效应分解
7. 用 GWR 检验空间异质性
8. 输出一份完整的分析报告（文字 + 图表）

### 输出

- 至少 6 张分析图
- 一段结论性文字（300字以内），回答：中国地级市经济发展水平存在怎样的空间模式？用什么模型最合适？主要发现是什么？

---

## 环境要求

```bash
pip install numpy pandas matplotlib geopandas
pip install libpysal esda spreg mgwr
pip install statsmodels scipy seaborn
```

## 注意事项

- Week 8 的数据已复制到 `data/china/` 目录下
- 所有分析使用 2021 年截面数据
- 注意处理缺失值和直辖市匹配问题（参考 Week 8 代码）
- 图表使用中文标题，`plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']`
