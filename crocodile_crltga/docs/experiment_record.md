# Experiment Record

本文档记录当前仓库公开保留的主实验结论，只保留真正影响最终模型选择的内容。

## 1. Task Setup

- 数据集：CheXpert 5-class protocol
- 标签：
  - `Atelectasis`
  - `Cardiomegaly`
  - `Consolidation`
  - `Edema`
  - `Pleural Effusion`
- domain 标签：`Frontal / Lateral`
- 不确定标签策略：`U-Ones`

## 2. Baseline Check

在进入融合模型调试前，先确认数据、环境和评估流程正常：

- 参考 baseline：DenseNet121 pretrained
- 同一 5 类协议下可以稳定训练
- 说明当前项目主要问题不在数据损坏，而在融合结构的训练稳定性

## 3. Main Failure Modes Observed

完整结构训练时，主要问题集中在数值稳定性：

- 训练早期出现 non-finite gradient
- 最初失败点位于 backbone 早层卷积
- 冻结 backbone 后，失败点转移到 domain feature block
- 继续冻结后，失败点转移到 disease feature block

这说明问题不是单一模块错误，而是复杂训练链路整体过于激进。

## 4. Stabilization Steps

最终保留下来的稳定化操作包括：

1. 冻结 backbone 到 `layer4`
2. 冻结对应 BatchNorm 统计
3. 对 domain 输入执行 `detach`
4. 冻结整个 domain branch
5. 旁路复杂 `CrocodileFeatureBlock` 路径，只保留轻量投影
6. 关闭 `triplet` 与 `dag`
7. 保留疾病主头和疾病干预辅助头
8. 训练时使用 AMP、梯度裁剪和梯度累积

## 5. Final Selected Configuration

最终采用配置：

- [`configs/chexpert_small_5label_stage2_evalfix_bs24_lr2e5_20e_diseaseonly.json`](../configs/chexpert_small_5label_stage2_evalfix_bs24_lr2e5_20e_diseaseonly.json)

关键配置：

- `batch_size = 24`
- `epochs = 20`
- `lr = 2e-5`
- `pretrained = true`
- `embedding_dim = 256`
- `transformer_layers = 2`
- `transformer_heads = 4`
- `lambda_y_sp = 0.0`
- `lambda_y_bd = 0.2`

## 6. Final Result Used in README

最终选用的是最佳单模型结果，而不是 checkpoint soup：

- `macro_auc = 0.8363661866572223`
- `tuned_macro_f1 = 0.6446668301182221`

说明：

- `weighted soup` 略微提高了 AUC
- 但它属于推理阶段的 checkpoint 融合
- 不对应新的网络结构，因此不作为主方法结构写入 README

## 7. Conclusion

当前公开版本的核心结论是：

- 原始全量因果融合结构在当前条件下不稳定
- 稳定可复现的主线是 `disease-only` 简化版本
- 该版本保留了 query 建模和因果干预思路
- 在公开仓库中，应该把它作为最终方法实现来展示
