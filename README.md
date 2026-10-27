# House Prices：第三版，全记录地区模型融合

本版继续保留全部 **1460 条训练记录和原始售价**。没有删除样本、裁剪售价、
显式设置样本权重或按测试 ID 手工指定预测值。测试集 1459 条记录全部输出。

上一版的实际 Kaggle 成绩为 **0.11883**。本版尚无线上成绩；下面的验证分数
不能当作已经取得的 Kaggle 成绩，也不能保证一定达到 0.11。

## 最终方案

按 log1p 预测进行以下固定融合：

- **35% 地区交互 Lasso**：学习各地区与面积、品质、房龄的中心化交互。
  面积上限只在训练折内估计，不再给所有数值特征统一截尾。
- **25% 地理加权 RBF-SVR**：在普通特征之外，额外加入放大 16 倍的地区及
  地区×交易类型独热特征，使核距离更重视这些差异。放大的是特征列，不是样本权重。
- **40% 新版 CatBoost**：depth=6、2800 轮、learning_rate=0.03、l2=3。
  使用紧凑原生类别及逐行面积、房龄、浴室等特征，减少冗余品质多项式。

训练目标为 `log1p(SalePrice)`，最后使用 `expm1` 还原成价格。
最终提交来自各模型在全部 1460 行上重训后的预测。

## 复现

已验证 Python 3.12.14，CPU，无需 GPU、Kaggle API 凭据或外部数据。
下载并解压 `house_prices_source2926.zip` 后，进入源码目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_v3.py --data-dir data --output-dir results_v3
```

默认重新计算 seed=42、2026、1109 三组五折验证，再进行全量重训并生成
`results_v3/submission2926.csv`。具体耗时取决于机器性能。

只重新训练并生成 CSV：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train_v3.py --skip-cv --output-dir fresh_results
```

复用包内已计算的 OOF，可运行：

```bash
python train_v3.py --reuse-oof v3_experiments --output-dir cached_results
```

缓存使用源码、数据、模型参数、库版本和产物 SHA-256 校验，且会核对全部 ID
和可用的折成员信息；更改源文件后须重新计算相应验证。

## 验证方法

- 评分严格使用 `sqrt(mean((log(真实售价)-log(预测售价))**2))`。
- 每一组 OOF 均覆盖原始全部 1460 条训练记录，包括两条特殊的大面积低价交易。
- 地区阈值、中心值、缺失填补、偏态变换、缩放和独热编码均在训练折内拟合。
- seed=42、2026 用于本轮开发与选择；确定模型和上述四舍五入权重后，
  再用新 seed=1109 比较上一版和本版。
- 这些仍是同一数据集的重新划分，不是独立外部测试集。
- 模型未使用测试集目标或外部价格数据；`Id` 只用于输出与对齐。

## 产物

- `submission2926.csv` / `results_v3/submission2926.csv`：新版提交文件。
- `results_v3/validation_report.json`：三组指标、折结果、版本、哈希和限制。
- `results_v3/oof_seed*.csv`：逐行 OOF，可自行复算所有验证分数。
- `results_v3/prediction_components.csv`：全量重训的各模型预测。
- `v3_comparison.json`：新旧版本在相同划分下的比较。
- `v3_experiments/`：原始实验指标与选中模型的 OOF 缓存。
- `data/`：用户提供的原始竞赛数据。

源码入口为 `train_v3.py`；三个模型分别位于 `v3_linear.py`、`v3_kernel.py`、
`v3_cat.py`。共用特征和验证模块也包含在包内。

`v3_boost.py` 是辅助残差提升及原生类别 LightGBM 实验，收益不足，未纳入提交；
运行前安装 `requirements-experiments.txt`。
CatBoost 的 Ordered/Huber 配置虽在实验源码中提供，但本轮未完整执行，未计入结论。

## 输出检查

CSV 应为 1459 行、两列 `Id,SalePrice`，不含额外索引；ID 顺序与测试集一致，
价格为正且有限。报告记录提交 SHA-256，可区分旧版文件和本版文件。

## 本轮实测比较

| 全量五折划分 | 上一版 V2 | 新版 V3 | 用途 |
|---|---:|---:|---|
| seed=42 | 0.118094 | 0.114449 | 开发选择 |
| seed=2026 | 0.118077 | 0.112920 | 开发选择 |
| seed=1109 | 0.115685 | 0.113471 | 固定方案后复核 |

所有分数均对全部 1460 条记录评分。三组都属于本地验证，新版线上分数尚待实际提交确认。

本轮两次全量重训生成的 CSV 逐字节一致；源码、数据和 OOF 校验已完成。
