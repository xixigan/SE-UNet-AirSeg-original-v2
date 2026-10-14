# SE-UNet-AirSeg 评价指标说明

本项目的评价指标体系采用 **ATM 22 挑战赛官方口径**，核心实现位于 `metrics.py`，
由 `train.py` 中的 `evaluation_case()` 在每个病例上调用，验证阶段对全部病例取**均值 ± 标准差**。

## 一、六个主指标

| 缩写 | 中文名 | 计算方式（代码公式） | 含义 |
|------|--------|----------------------|------|
| TD | 树检测率 (Tree Detection) | `Σ(P∩S) / Σ(S) × 100`，S 为金标准骨架 | 骨架体素被预测覆盖的长度比例；召回导向，过预测容易刷高 |
| BD | 分支检出率 (Branch Detected) | 每条分支被覆盖比例 ≥ 0.8 记为检出；`检出分支数 / 总分支数 × 100` | 按 parsing 标签逐分支统计的检出率 |
| DSC | Dice 相似系数 | `2·Σ(P∩T) / (ΣP + ΣT) × 100` | 体素级重叠度，最严格的分割质量指标 |
| Pre | 精确率 (Precision) | `TP / ΣP × 100`，其中 TP = Σ(P∩T) | 预测体素中真阳占比；过预测会直接拉低该项 |
| Sen | 敏感度 (Sensitivity) | `100 − FNR`，FNR = `Σ(T − P∩T) / ΣT × 100`，等价于 `TP / T × 100` | 真阳召回率 |
| Spe | 特异度 (Specificity) | `100 − FPR`，FPR = `Σ(P − P∩T) / Σ(1−T) × 100`，等价于 `TN / 背景 × 100` | 真阴占比；背景体素巨大，数值虚高、区分力弱 |

符号约定：P = 预测掩码，T = 金标准 label，S = 金标准骨架。
所有公式带平滑项 `smooth = 1e-5` 防止除零。

## 二、混淆矩阵基础量

以体素为单位，预测 P 与金标准 T 交叉形成四个量：

| 名称 | 定义 | 临床类比 |
|------|------|----------|
| TP 真阳 | 预测有、金标准也有 (`P∩T`) | 正确查出的病灶 |
| FP 假阳 | 预测有、金标准没有 (`P − P∩T`) | 误诊 |
| FN 假阴 | 金标准有、预测没有 (`T − P∩T`) | 漏诊 |
| TN 真阴 | 两者都没有 (`1 − P − T + P∩T`) | 正确排除的健康区 |

由此派生：

- **FPR 假阳率（误报率）** = `FP / (FP + TN)` = FP / 全部背景体素。本项目中 `Spe = 100 − FPR`，FPR 不单独输出。
- **FNR 假阴率（漏检率）** = `FN / (FN + TP)` = FN / 全部真值体素。`Sen = 100 − FNR`。

## 三、Sen 与 Pre 的互补关系

| | Sen（敏感度/召回率） | Pre（精确率/查准率） |
|---|---|---|
| 回答的问题 | 真实气道有多少被找出来？ | 预测出来的体素有多少是真的？ |
| 公式 | `TP / (TP + FN)` | `TP / (TP + FP)` |
| 分母 | 金标准总体素数 ΣT | 预测总体素数 ΣP |
| 惩罚的错误 | 漏检（FN） | 误检（FP） |
| 拉高方式 | 大胆多预测 | 保守少预测 |
| 副作用 | Pre 下降 | Sen 下降 |

两者天然互相牵制。气道分割中背景体素占 99% 以上，FPR/Spe 的分母巨大，
多预测几千个假阳体素对 Spe 几乎无影响（虚高、区分力弱）；
真正约束过预测的是 **Pre**——它的分母是预测总体积，对假阳敏感得多。

## 四、为什么医学术语用 Sensitivity/Specificity 而非 Recall/Precision

**本质上是同一组公式，只是命名传统不同：**

```
Sen (敏感度) = TP/(TP+FN) ≡ 召回率 (Recall)
Pre (精确率) = TP/(TP+FP) ≡ 查准率 (Precision)
```

医学坚持传统命名的原因：

1. **历史源头不同**。Sen/Spec/FPR/FNR 源自 1940-50 年代的诊断试验与流行病学
   （ROC 曲线由雷达信号检测理论移植而来），思维框架是"患病 vs 健康"的 2×2 列联表；
   Precision/Recall/F1 源自 1970-80 年代的信息检索（IR），后被机器学习社区沿用。
   两套词汇平行发展，各自进入教科书与期刊规范（如 STARD 报告指南要求临床试验
   必须报告 sensitivity/specificity）。

2. **特异性有直接临床意义**。IR 中"非相关文档"无人关心；医学里 Specificity 量化
   误诊率——健康人被误判为患病即假阳，因此 Sen/Spec 是医学的天然对偶。

3. **分母更直观**。Sensitivity 分母 = 所有患者（"100 个患者查出几个"），
   Specificity 分母 = 所有健康人（"100 个健康人排除几个"），都是客观存在的群体，
   便于向患者解释；而 Precision 的分母依赖模型自身输出量。

现代医学影像 AI 论文中两套术语已混用为常态。本项目 ATM22 的综合评分公式
同时使用 Pre（ML/IR 词汇）与 Sen（医学词汇），正是两个社区融合的痕迹。

## 五、计算前的关键后处理

`evaluation_case()` 在计算指标前，先用 `cc3d.connected_components(pred, connectivity=26)`
做 26 连通域标记，**只保留最大连通域**（`large_cd`）再计算六个指标。

依据：气道是一棵连通的树，孤立碎块必为假阳；ATM22 官方口径同样如此规定。
（26 连通比 6 连通宽松，对角相邻也算连通。）

## 六、综合评分（选择最佳 epoch）

`weight_br.py` 的 `valid_recall()` 从日志解析六项指标后，按官方加权公式打分：

```
SCORE = (TD + BD) × 0.15 + (DSC + Pre) × 0.2 + Sen × 0.3
```

取 SCORE 最大的 epoch 作为最佳模型。注意 Spe（即 100 − FPR）**不计入 SCORE**，
原因见第三节：背景巨大导致其区分力弱。

## 七、训练期损失函数（非评价指标）

| 名称 | 位置 | 公式 / 说明 |
|------|------|-------------|
| dice_loss | `train.py` | 标准 Dice 损失：`1 − 2·|P∩T| / (|P| + |T|)` |
| Tversky_loss / general_union_loss | `train.py` | 加权 Tversky（stage1 α=0.1、stage2 α=0.2），可不对称惩罚 FP / FN |
| break_loss | `train.py` | 断点损失，stage2 难例遍使用 |
| 整卷 Dice 打印 | `save_gradients.py`、`weight_br.py` | 滑窗推理时的粗检 `2·Σ(P∩T)/(ΣP+ΣT)`，未做最大连通域后处理，口径与验证指标不同，仅供观察 |

## 八、相关源文件

- `metrics.py`：六个指标的具体实现（`tree_length_calculation`、`branch_detected_calculation`、`dice_coefficient_score_calculation`、`precision_calculation`、`sensitivity_calculation`、`specificity_calculation`、`false_positive_rate_calculation`、`false_negative_rate_calculation`）
- `train.py`：`evaluation_case()` 单病例评估、`validation()` 验证集汇总与日志输出
- `weight_br.py`：`valid_recall()` 综合评分选最佳 epoch
