# Architecture Overview

本文档只描述当前仓库最终保留、并实际用于课程项目主实验的网络结构，不再回顾已放弃的中间结构版本。

## 1. Final Model Choice

最终采用的是一个稳定化后的 `disease-only` 版本，配置文件为：

- [`configs/chexpert_small_5label_stage2_evalfix_bs24_lr2e5_20e_diseaseonly.json`](../configs/chexpert_small_5label_stage2_evalfix_bs24_lr2e5_20e_diseaseonly.json)

核心目标是保留 CROCODILE 的因果干预思想与 CRLTGA 风格的 query 表示建模，同时删除在当前数据和硬件条件下不稳定的训练路径。

## 2. Pipeline

整体前向流程如下：

1. 输入 CheXpert 胸片，统一缩放到 `224 x 224`
2. `ResNet50` backbone 输出共享特征图 `2048 x 7 x 7`
3. 疾病分支和领域分支分别经过 `CrocodileFeatureBlock`
4. 加入二维位置编码后送入 `TokenTransformerEncoder`
5. `TaskQueryGenerator` 从 token 序列中抽取查询表示 `Q` 与对照表示 `Q_bar`
6. `CausalDisentangler` 对查询特征做因果/伪相关解耦
7. 对 `Q_bar` 做 batch 内打乱并与 `Q` 相加，形成干预表示
8. 通过 group-wise heads 输出疾病主预测、伪相关预测和干预预测

## 3. Module Responsibilities

### 3.1 `src/models/backbone.py`

- `ResNetBackbone`
- 调用 `torchvision.models.resnet50`
- 提供共享卷积特征
- 支持按层冻结，当前主线冻结到 `layer4`

### 3.2 `src/models/crocodile_blocks.py`

- `CrocodileFeatureBlock`
- `CausalityMapBlock`
- 提供 feature projection、轻量旁路和因果图辅助输出
- 当前主线开启 `bypass_complex_feature_block`

### 3.3 `src/models/transformer.py`

- `TokenTransformerEncoder`
- 将 `7 x 7` 特征图展平为 token 序列并建模全局依赖

### 3.4 `src/models/heads.py`

- `TaskQueryGenerator`
- `GroupWiseLinear`
- `GroupWiseLinearAdd`
- 负责 query 抽取和多标签头部映射

### 3.5 `src/models/disentangler.py`

- `CausalDisentangler`
- `batch_triplet_loss`
- `dag_penalty`
- 负责 query 级因果/伪相关拆分，以及可选结构正则

### 3.6 `src/models/network.py`

- `CrocodileCrltgaNet`
- 负责把 backbone、双分支、query、解耦、干预和输出头连接成完整网络

### 3.7 `src/train.py`

- 数据变换、DataLoader、损失函数、训练循环、验证、checkpoint 保存
- 支持 `best.pt`、`best_auc.pt`、`best_tuned_f1.pt`

## 4. Final Training Behavior

最终采用版本不是完整双任务联合优化，而是稳定化后的主任务版本：

- `freeze_backbone_until = layer4`
- `freeze_disease_feature_block = true`
- `detach_domain_feature_map = true`
- `freeze_domain_branch = true`
- `bypass_complex_feature_block = true`
- `lambda_d_main = 0`
- `lambda_d_sp = 0`
- `lambda_d_bd = 0`
- `lambda_triplet = 0`
- `lambda_dag = 0`

因此，虽然实现中保留了 domain branch 和结构损失接口，但最终主实验真正优化的是：

- 疾病主预测 `z_x`
- 疾病干预辅助预测 `z_c_cap`

## 5. Why This Version Was Kept

选择这一版本的原因很直接：

- 完整结构在当前设置下存在明显数值不稳定
- 冻结 backbone 和 domain branch 后，训练可稳定完成
- 保留了 query-level 建模与干预头，仍能体现方法设计思路
- 在 CheXpert 5 类验证集上取得了当前仓库中最可靠的单模型结果
