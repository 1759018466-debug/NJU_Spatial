# 编程实践题2：空间权重矩阵构建与空间回归模型（中等难度）

## 任务描述

使用中国地级市数据，完成从空间权重矩阵构建到空间回归模型估计的完整分析流程。

## 分析步骤

1. **数据加载**：加载 shapefile 和 CSV，筛选 2021 年数据，处理直辖市映射，合并
2. **空间权重矩阵**：构建 Queen 邻接 + KNN(4) + KNN(8)，行标准化，可视化连接图
3. **OLS 基准模型**：使用 statsmodels 估计经典线性回归
4. **空间回归模型**（使用 `spreg`）：
   - SAR（空间滞后模型 `ML_Lag`）
   - SEM（空间误差模型 `ML_Error`）
   - SDM（空间杜宾模型 `ML_Error_Lag`）
5. **LM 检验**：LM-Lag、LM-Error、Robust LM-Lag、Robust LM-Error，判断模型选择
6. **模型比较**：用表格对比 R²、AIC、Log-Likelihood、各参数
7. **残差 Moran's I**：对比 OLS 和空间模型的残差空间自相关是否消除

## 数据路径

- 城市边界：`data/china/city.shp`
- 属性数据：`data/china/地级市数据.csv`

## 运行方式

```bash
cd exercise_02_spatial_models/
python spatial_models.py
```

## 依赖库

```
numpy pandas matplotlib geopandas libpysal spreg statsmodels esda scipy seaborn
```

## 输出文件

| 文件名 | 内容 |
|--------|------|
| `output_01_weight_connections.png` | 三种权重矩阵连接图 |
| `output_02_ols_summary.png` | OLS 回归结果表 |
| `output_03_lm_tests.png` | LM 检验结果可视化 |
| `output_04_model_comparison.png` | 模型比较汇总表 |
| `output_05_residual_moran.png` | 残差 Moran's I 对比 |
| `output_06_sdm_coefficients.png` | SDM 模型系数解读图 |
