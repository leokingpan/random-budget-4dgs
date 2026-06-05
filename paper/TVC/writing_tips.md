# TVC 期刊写作技巧总结——基于 SpatioGS 与 CLIPGaussian 的对比分析

> 两篇论文：
> - **SpatioGS** — Cao et al., The Visual Computer, Vol.42, Article 215 (2026). 动态GS密度控制。
> - **CLIPGaussian** — Cao et al., The Visual Computer, Vol.41, pp.5815-5833 (2025). GS语义引导渲染，开放获取。

---

## 一、摘要写作模板（TVC 黄金结构）

SpatioGS 的摘要结构是 TVC 论文的标准范式，逐句拆解如下：

| 句序 | 功能 | SpatioGS 示例 | 你的论文对应位置 |
|:---:|------|--------------|--------------|
| 1 | 领域背景 | "3D Gaussian Splatting technique has achieved remarkable results in scene rendering." | 4DGS 是动态场景重建的重要进展 |
| 2 | 现存问题 | "However, it still struggles to represent dynamic scenes with complexity..." | 模型基元数量固定，无法适配多设备 |
| 3 | 方法引入 | "To overcome these challenges, we introduce SpatioGS, an innovative method..." | 提出随机预算训练策略 |
| 4-6 | **三项贡献逐一列举** | "First, we propose... Secondly, we incorporate... Thirdly, we propose..." | 三项贡献（拓展+评分+损失设计） |
| 7-8 | 实验验证+关键数字 | "We assess... on... datasets... Experimental results demonstrate... achieves 86 FPS..." | 9场景+2.0 dB +47% FPS +8.2%开销 |

**关键发现**：TVC 论文的摘要中，**贡献以 "First... Secondly... Thirdly..." 的形式在摘要正文中直接列举**，而非仅在文末的贡献列表中。这说明 TVC 审稿人期望在摘要中就能看到清晰、可数的技术贡献。

---

## 二、引言写作结构

SpatioGS 和 CLIPGaussian 的引言均遵循五段式结构：

| 段落 | 内容 | 要点 |
|:---:|------|------|
| 1 | 问题的重要性 + 现有方法的进展 | 不要从"远古"开始——直接从 3DGS/NeRF 切入 |
| 2 | 现有方法的**具体**局限 | 不要泛泛说"有问题"，要说清楚"什么问题、为什么是问题" |
| 3 | 我们的方法（高层概述） | "We propose... / We introduce..." 一句话说清核心思路 |
| 4 | 技术难点 + 我们如何解决 | 这是显示技术深度的段落——说出别人没解决的问题，以及你的方案 |
| 5 | 关键结果 + 贡献列表 | 数字先行（"+2.0 dB"），再跟贡献项 |

---

## 三、措辞技巧对照

### 3.1 动词选择

| 避免（弱/口语化） | 推荐（TVC 风格） |
|---|---|
| "We try to..." | "We address/investigate..." |
| "Our method can..." | "Our method achieves/demonstrates..." |
| "We think that..." | "Our experiments confirm/suggest..." |
| "It is important to..." | "This is critical for..." / 直接陈述 |
| "This paper wants to..." | "This paper investigates/proposes..." |

### 3.2 转折与递进

| 场景 | TVC 常用表达 |
|------|-------------|
| 引出问题 | "However, ... still struggles to..." |
| 过渡到方法 | "To overcome these challenges, we introduce..." |
| 补充说明 | "We further investigate... / In addition..." |
| 强调重要性 | "This is particularly important when..." |

### 3.3 贡献表述（避免过于谦虚）

| 过于谦虚 | TVC 风格 |
|---|---|
| "初步验证了..." | "Our experiments confirm/demonstrate that..." |
| "可能具有一定优势" | "achieves a consistent improvement" |
| "仍需进一步验证"（在结论中） | 放在 Limitations 小节中具体讨论 |
| 贡献以 itemize 列表呈现 | 同时在摘要正文和 Introduction 末尾出现 |

### 3.4 数量词的表述

TVC 论文中，关键数字**贯穿全文反复出现**，形成记忆锚点：

- 摘要中："+2.0 dB, 9 scenes, +47%, +8.2% overhead"
- Introduction 末尾：再次出现
- 各节正文：数据分析时再次引用
- Conclusion：第三次出现

这不是冗余——这是学术写作中的 "signal reinforcement" 策略。

---

## 四、章节结构对比

### SpatioGS（10页, 11图 + 1算法）

```
1 Introduction (含贡献列表)
2 Related Work (不分子节——合并为一个节)
3 Method
  3.1 概述
  3.2 三个子模块依次展开
4 Experiments
  4.1 实验设置
  4.2 与SOTA对比 (2个数据集，2张表)
  4.3 消融实验
  4.4 可视化
5 Conclusion
```

### CLIPGaussian（19页）

```
1 Introduction (含编号贡献列表 (1)(2)(3))
2 Related Works (分为4个小节)
3 Method (3.1 基础回顾, 3.2 方法详述, 3.3 评估方法)
4 Experiments
5 Discussion (含 Limitations)
6 Conclusion
```

### 你的论文（当前21页）

```
1 Introduction ✓（含贡献列表）
2 Related Work（3小节）✓
3 Method（5小节）✓
4 Experiments（5小节）✓
5 Discussion（含Limitations）✓
6 Conclusion ✓
```

**结构上与TVC范式一致**，不需要调整。

---

## 五、TVC 论文的 "诚实度" 特征

从 SpatioGS 和 CLIPGaussian 中可以观察到一个共同特点：**TVC 接受一定程度的诚实自我批评**，但批评需要有具体分析而非泛泛而谈。

| 不好的局限表述 | 好的局限表述 |
|---|---|
| "Our method may not work well on some scenes." | "On the press scene, our method underperforms naive MGS-4D by 1.59 dB. This inversion suggests that in specular scenes, opacity alone may serve as a better ranking signal because the visual content is dominated by highlights where visibility outweighs spatial extent." |
| "Future work will improve this." | "Future work could incorporate color variance or spherical harmonic energy into the importance function to better capture appearance-critical primitives." |

**关键**：局限 → 分析原因 → 具体改进方向。三个环节缺一不可。

---

## 六、图表排版注意事项

从 SpatioGS（11图 + 1算法块，约10页）可以学到：

1. **定性对比图放在主结果附近**——SpatioGS 的渲染对比图紧跟在定量表格之后
2. **消融实验的图表紧密排列**——不在消融和主结果之间插入大段文字
3. **Overview 图放在实验部分开头**（而非 Introduction）——让读者先读到方法，再看全局对比
4. **每张图都有一段"导读"文字**——不是仅靠 caption 说明，正文中有对图的具体解读

---

## 七、标题/副标题的命名技巧

SpatioGS 的标题模式：`方法名: 技术特征 for 应用场景`

- `SpatioGS: spatiotemporal-aware density control for dynamic scene rendering with Gaussian splatting`

你的论文标题已经符合这个模式：

- `Stochastic Budget Training for 4D Gaussian Splatting under Variable Inference Budgets`

不需要修改。

---

## 八、关键词策略

SpatioGS 的关键词：`Gaussian splatting · Dynamic scene · Real-time rendering · Spatiotemporal-aware`

- 4个关键词，简洁覆盖：方法类型、任务类型、技术卖点、技术特征
- 不堆砌——不是越多越好

你的论文：`4D Gaussian Splatting, stochastic budget training, dynamic scene rendering, primitive importance scoring, variable-budget inference, level of detail`

- 6个关键词，稍多，可以考虑合并 "variable-budget inference" 和 "level of detail" 为一个
- 建议压缩为4-5个
