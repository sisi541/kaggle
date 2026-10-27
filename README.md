# House Prices：保留全部记录的第二版

本版训练和验证始终保留原始 **1460 条训练记录**，包括 ID 524、1299。
没有删除记录、裁剪售价或设置样本权重；测试集 1459 条记录全部输出。
入口是 `train_full.py`，交付文件为 `submission2926.csv`。

## 复现

已验证 Python 3.12.14，CPU，无需 GPU、Kaggle API 凭据或外部数据。
源码包包含原始竞赛数据。进入解压后包含本文件的目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_full.py --data-dir data --output-dir results_full
```

默认重新执行三组五折验证，再在全部 1460 条记录上重训生成 CSV。
只重训和生成提交文件：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_full.py --skip-cv --output-dir fresh_results
```

如需复用包内本轮已生成的两组 OOF，减少计算：

```bash
python train_full.py --reuse-oof full_experiments --output-dir cached_results
```

缓存会核对源码、原始数据、权重和 OOF 文件的 SHA-256，并核对全部训练/测试 ID。
未提供 `--reuse-oof` 时，默认从原始 CSV 重新验证。默认同时计算 seed 2926 的新 OOF。

## 最终模型

对数空间融合：

```text
0.35 × Lasso（按 Neighborhood 学习折内面积上限）
+ 0.30 × SVR（折内数值特征截尾）
+ 0.35 × CatBoost（原生类别特征）
```

每个模型训练目标为 `log1p(SalePrice)`；最终用 `expm1` 还原售价。
Lasso/SVR 对输入特征的分位数处理仅在训练折内拟合，所有行和原始售价都保留。
CatBoost 使用完整原始记录以及逐行计算的面积、房龄、品质和地区交互特征。
所有模型均不使用 `Id` 预测，不为任何测试 ID 手工指定售价。

本次比较了用户建议的四个方向：Ridge/Lasso、Random Forest、XGBoost、CatBoost；
另保留 SVR 作为有互补作用的模型。随机森林和 XGBoost 在本轮融合中未提供额外改善。
各模型源码和实验指标均在包内，可独立复查。

## 验证与限制

- 每组均对 **1460 条记录** 产生 OOF 预测，按比赛的精确自然对数 RMSE 评分。
- seed 42、2026 用于开发比较；确定模型和四舍五入权重后，再计算 seed 2926。
- 三组划分仍来自同一个数据集，不能称为独立外部测试集。
- 两条特殊交易仍可能有较大误差；报告包含它们的 OOF 预测和所属折。
- 旧版本的实际 Kaggle 成绩为 **0.12375**。新版尚无 Kaggle 成绩，不能保证达到 0.11。
- 原版本的 0.10743 是剔除两条记录后的结果，不能与本版全记录验证直接比较。
  原线性融合在全 1460 条记录上的 seed42 结果为 0.12917。

完整数值见 `results_full/validation_report.json`；逐样本验证预测见
`results_full/oof_seed*.csv`，可直接复算分数。CSV 满足 `Id,SalePrice` 格式，
不带索引，测试 ID 顺序一致，预测全部为正且有限。

## 文件

- `train_full.py`：最终训练、完整验证、提交生成入口。
- `features.py`、`linear_models.py`：共用特征与折内预处理。
- `full_linear.py`、`full_linear_features.py`：全记录 Ridge/Lasso/SVR 实验与模型。
- `full_cat.py`：全记录 CatBoost 实验与 sklearn 兼容模型。
- `full_forest.py`：全记录 Random Forest/ExtraTrees 实验。
- `full_xgb.py`、`full_cv.py`：全记录 XGBoost 与共用验证器。
- `full_experiments/`：各方向指标、选中模型 OOF 及校验清单。
- `results_full/`：最终验证报告、OOF、基模型预测和提交文件。
- `data/`：用户提供的原始比赛数据。

辅助实验命令：

```bash
python full_linear.py --seed 42 --models full_lasso_neighborhood98 full_ridge_clip full_svr_clip
python full_forest.py --seed 42 --models rf_leaf1
python full_xgb.py --seed 42 --models xgb_d3
python full_cat.py --seed 42 --models cat_full_d5
```

## 本轮实测结果

| 模型 | seed42 五折 | seed2026 五折 |
|---|---:|---:|
| Lasso | 0.121573 | 0.122139 |
| Ridge | 0.123701 | 0.123204 |
| SVR | 0.122022 | 0.122982 |
| XGBoost | 0.127279 | 0.120031 |
| CatBoost | 0.123920 | 0.123178 |
| Random Forest | 0.140291 | 0.137833 |
| Final ensemble | 0.118094 | 0.118077 |

固定融合方案后，seed2926 五折复核为 **0.116778**。所有分数均覆盖全部 1460 条记录。

两次独立重训生成的提交 CSV 逐字节一致。以上为本地交叉验证，不是新版 Kaggle 成绩。
