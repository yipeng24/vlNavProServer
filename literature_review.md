# Literature Review / 文献综述

**Topic:** 无先验地图的长指令/模糊指令 VLN 导航方法：拓扑记忆 × 历史帧增强
**Topic (EN):** Mapless VLN under Long/Ambiguous Instructions with Topological Memory and Frame History

> 检索日期 / Search date: 2026-03-21 | 覆盖渠道 / Sources: arXiv, Web, Semantic Scholar

---

## 一、文献全表 / Complete Paper Table

### 1.1 无地图 VLN + 拓扑/场景记忆

| 论文 | 作者 | 年份 | 发表渠道 | 无预建图 | 拓扑记忆 | 模糊/长指令 | 帧历史 | 失败纠错 |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **MSNav**: Zero-Shot VLN with Dynamic Memory and LLM Spatial Reasoning | Liu et al. | 2025 | arXiv 2508.16654 | ✓ | ✓动态拓扑 | ✓长horizon | 部分 | ✗ |
| **EmergeNav**: Structured Embodied Inference for Zero-Shot VLN | Luo, Ma | 2026 | arXiv 2603.16947 | ✓ | ✓双记忆 | 部分 | 隐含 | ✗ |
| **MC-GPT**: Empowering VLN with Memory Map and Reasoning Chains | Zhan et al. | 2024 | arXiv 2405.10620 | 部分 | ✓拓扑图 | ✗ | ✓视点存储 | ✗ |
| **RAGNav**: Retrieval-Augmented Topological Reasoning for Multi-Goal VLN | Luo, Bai | 2026 | arXiv 2603.03745 | 部分 | ✓双基拓扑 | 多目标 | ✗ | ✗ |
| **MapNav**: Memory via Annotated Semantic Maps for VLN | Zhang et al. | 2025 | arXiv 2502.13451 | ✓ | ✗语义2D图 | ✗ | ✗(替代帧) | ✗ |
| **LIFGIF**: Zero-shot Object-Centric Instruction Following | Raychaudhuri et al. | 2024 | arXiv 2411.07848 | ✓ | ✓3D地标图 | ✓自然语言 | ✗ | ✗ |
| **InstructNav**: Zero-shot Generic Instruction Navigation | Long et al. | 2024 | arXiv 2406.04882 | ✓ | ✗ | 部分 | ✗ | ✗ |
| **ImagineNav**: VLMs as Navigator through Scene Imagination | Zhao et al. | 2024 | ICLR 2025 / 2410.09874 | ✓ | ✗ | 部分 | 部分 | ✗ |
| **OIKG**: Fine-Grained Instruction-Guided Graph Reasoning for VLN | Liu et al. | 2025 | arXiv 2503.11006 | ✗(需导航图) | ✓导航图 | ✓密集长指令 | ✗ | ✗ |
| **SE-VLN**: Self-Evolving VLN Framework Based on MLLMs | Dong et al. | 2025 | arXiv 2507.13152 | 隐含 | ✗分层情节记忆 | ✗ | ✗ | ✗ |
| **Mobility VLA**: Multimodal Instruction Navigation with Topological Graphs | Chiang et al. | 2024 | arXiv 2407.07775 | ✗(预建拓扑图) | ✓ | ✓模糊多模态 | ✓演示视频帧 | ✗ |

### 1.2 帧历史 / 时序上下文

| 论文 | 作者 | 年份 | 发表渠道 | 时序机制 | 关键设计 | 真实机器人 |
|---|---|---|---|---|---|:---:|
| **NaVid**: Video-based VLM Plans the Next Step | Zhang et al. | 2024 | RSS 2024 / 2402.15852 | 全轨迹帧序列输入VLM | 仅单目RGB，无地图/深度/里程计 | 有限 |
| **VLN-R1**: VLN via Reinforcement Fine-Tuning | Qi et al. | 2025 | arXiv 2506.17221 | Long-Short Memory Sampling | 近期帧与远期帧均衡采样 + RL微调 | ✗ |
| **PROSPECT**: Unified Streaming VLN | Fan et al. | 2026 | arXiv 2603.03739 | 流式查询Token融合 | 语义-空间特征流式融合；真实机器人验证 | ✓ |
| **History-Conditioned Token Pruning** for Efficient VLN | Wang et al. | 2026 | arXiv 2603.06480 | 时空Token压缩 | 基于注意力的历史token重要性剪枝 | ✓(Unitree Go2) |
| **Efficient-VLN**: Training-Efficient VLN | Zheng et al. | 2025 | arXiv 2512.10310 | 渐进+递归记忆 | 近期帧分配更多Token；KV-cache递归记忆 | ✗ |
| **LH-VLN**: Towards Long-Horizon VLN | Song et al. | 2024 | CVPR 2025 / 2412.09082 | 多粒度动态记忆(MGDM) | 短时模糊+长时检索，150步导航基准 | ✗ |
| **ESceme**: VLN with Episodic Scene Memory | Zheng et al. | 2023 | IJCV 2024 / 2303.01032 | 情节场景记忆 | 回访已知场景时调用历史记忆 | ✗ |
| **StreamVLN**: Streaming VLN via SlowFast Modeling | Wei et al. | 2025 | arXiv 2507.05240 | 显式滑动窗口 | 快速响应窗口+慢速历史压缩记忆 | ✗ |

### 1.3 失败恢复与回退纠错

| 论文 | 作者 | 年份 | 发表渠道 | 纠错机制 | 触发条件 | 真实机器人 |
|---|---|---|---|---|---|:---:|
| **SmartWay**: Enhanced Waypoint Prediction and Backtracking | Shi et al. | 2025 | IROS 2025 / 2503.10069 | MLLM历史感知推理 + 自适应回退 | 路径失败检测 | ✓(Turtlebot 4) |
| **CorrectNav**: Self-Correction Flywheel for VLA Navigation | Yu et al. | 2025 | arXiv 2508.10416 | 错误轨迹自动生成纠正训练样本 | 在线自我纠错飞轮 | ✓(室内外真机) |
| **EventNav**: VLN via Event Knowledge Graph | Zhao et al. | 2024 | CIKM 2024 / 2408.02535 | 动态历史回溯模块 | 动作规划错误累积 | ✗ |
| **E²BA**: Exploration and Backtracking Agent | Shi et al. | 2025 | IEEE TCSVT / 2311.00530 | 双级联回溯判别器 | 导航停滞检测 | ✗(仿真) |
| **GC-VLN**: Instruction as Graph Constraints | Yin et al. | 2025 | CoRL 2025 / 2509.10454 | 约束图回溯 | 约束不满足时导航树回退 | ✗ |
| **VLM Failure Recovery** with Optimized Prompts | Chen et al. | 2024 | arXiv 2409.03966 | VLM提示优化纠错 | 失败检测+恢复计划生成 | ✗(乐高实验) |
| **CA-Nav**: Constraint-Aware Zero-Shot VLN | Chen et al. | 2024 | arXiv 2412.10137 | 约束感知子指令切换 | 子指令进度追踪 | ✓(真实机器人) |
| **MAG-Nav**: Memory-Reserved Active Grounding | — | 2025 | arXiv 2508.05021 | 视角主动定位+记忆回退 | 视觉歧义检测 | ✗ |

### 1.4 模糊/长指令处理

| 论文 | 作者 | 年份 | 发表渠道 | 指令类型 | 处理策略 | 真实机器人 |
|---|---|---|---|---|---|:---:|
| **Resolving Positional Ambiguity** in Dialogues by VLMs | Chen et al. | 2024 | arXiv 2410.12802 | 位置指代歧义 | 多轮对话消歧 | ✓(ROS机器人) |
| **Ask-to-Clarify**: Resolving Instruction Ambiguity | Lin et al. | 2025 | arXiv 2509.15061 | 主动请求澄清 | VLM对话+扩散模型动作 | ✓(8类真实任务) |
| **Mind the Error!**: VLN Instruction Error Detection | Taioli et al. | 2024 | IROS 2024 / 2403.10700 | 指令错误/歧义检测 | 跨模态Transformer错误检测器 | ✗ |
| **VELMA**: LLM Agents for VLN in Street View | Schumann et al. | 2024 | AAAI 2024 / 2307.06082 | 城市街景欠指定指令 | LLM文本化视觉观测 + CLIP地标提取 | ✗ |
| **SayNav**: LLMs for Dynamic Planning to Navigation | Rajvanshi et al. | 2023 | ICAPS 2024 / 2309.04077 | 多目标开放词汇 | 3D场景图 + LLM动态子目标分解 | ✗ |
| **CA-Nav** | Chen et al. | 2024 | arXiv 2412.10137 | 长指令子指令分解 | 约束感知完成追踪 | ✓ |
| **NaVILA**: Legged Robot VLA Navigation | Cheng et al. | 2024 | arXiv 2412.04453 | 长指令分解 | VLM→语言子目标→RL执行器 | ✓(legged) |
| **SLAM-Free Hierarchical VLN** | — | 2025 | arXiv 2509.20739 | 长指令分层 | 场景级+目标级VLM+LLM拓扑规划 | ✗ |

---

## 二、主题综述 / Thematic Synthesis

### 主题一：无地图 VLN 的记忆表示

无地图 VLN 的核心困难是**如何在不依赖预建图的情况下维持空间记忆**。现有工作呈现出三条路线：

**路线 A — 视频/帧历史直接输入 VLM（NaVid, VLN-R1, PROSPECT）**：将过去所有或部分 RGB 帧编码为时序上下文，直接输入大型视觉语言模型。优点是结构简单，摆脱所有传感器依赖；缺点是 token 代价随 episode 长度增长，且帧序列本身缺乏结构化的空间索引能力。

**路线 B — 在线构建语义/拓扑地图（MSNav, MC-GPT, RAGNav, MapNav）**：在导航过程中动态构建一个轻量级的地图结构，代替原始帧历史作为 VLM 的上下文。MSNav（2025）使用动态拓扑地图 + 节点剪枝，是本路线最新代表；MapNav（2025）则以带文字标注的俯视语义地图取代帧，代价较低但丢失时序信息。

**路线 C — 混合记忆（EmergeNav, StreamVLN, Efficient-VLN）**：同时维护短时高分辨率记忆（近期帧）与长时压缩记忆（历史摘要/地图），是当前精度-效率权衡的前沿趋势。

**与 NavLogic 的关系**：NavLogic 的 iLTM 属于路线 B 的变体，但更轻量——不维护度量地图或栅格，仅存储决策节点和地标节点的拓扑关系，配合路线 A 的有界帧历史（ILGP 滑动窗口），形成 A+B 的组合，尚无直接先行工作。

---

### 主题二：失败恢复与回退机制

导航失败恢复是 VLN 中一个被低估的问题。IROS 2025 的 **SmartWay** 是目前最相关的工作：它在零样本 VLN-CE 框架中实现了"检测路径失败 → 历史感知推理 → 自适应回退"的完整循环，并在真实 Turtlebot 4 上验证了碰撞率的下降（0.067→0.044）。

CoRL 2025 的 **GC-VLN** 将指令分解为有向无环图约束，当约束不满足时触发导航树回退，是形式化最完整的回退机制之一，但缺乏语义层面的失败判断（只依赖物理传感器）。

**CorrectNav**（2025）采用自我纠错飞轮，将错误轨迹自动转化为训练样本，是在线学习视角的纠错方法。

**关键差异**：上述工作的失败触发信号均来自物理传感器（碰撞、路径规划失败），或需要额外训练。NavLogic 提出的**故障传播式纠错**将 VLM 的语义失败判断（`noway`）作为纠错的一级触发信号，与物理传感器信号并行融合，不需要额外训练，且对语义死路（门虽开但视觉上无法导航）有独特响应能力。

---

### 主题三：模糊与长指令处理

现有方法处理模糊指令的策略分为三类：

1. **主动对话消歧**（Ask-to-Clarify 2025, Resolving Positional Ambiguity 2024）：通过与人类多轮对话消除歧义。适合人在环场景，但要求人类随时在场，不适合全自主部署。

2. **指令错误检测**（Mind the Error! IROS 2024）：在执行前检测指令中的错误/歧义。这是预处理视角，不处理执行中遇到的环境不一致。

3. **语义目标分层理解**（E²BA 2025, CA-Nav 2024, SayNav 2024）：将指令分解为可验证的子目标，当某子目标不可达时切换策略。CA-Nav 的约束感知完成追踪（CSM）与 NavLogic 的双粒度指令分层最为接近，但 CA-Nav 针对连续 VLN-CE 基准，不涉及室内外过渡的开放环境。

**NavLogic 的差异**：NavLogic 提出**过程指令可旁路、目标指令必须满足**的双粒度分层，这是目前文献中缺失的一个精细设计——现有方法要么把所有子目标都视为必须完成（CA-Nav），要么在模糊目标时依赖人类澄清（Ask-to-Clarify）。

---

### 主题四：帧历史的设计选择

以 VLMnav（2024）的消融为警示：K=5/10/15 帧历史窗口在他们的系统中**未能提升性能**，因此最终放弃。这一结果值得注意，需要在 NavLogic 中给出正面回应：

- VLMnav 的视觉输入格式是"带编号箭头的 overlay 图像"，这种格式本身就削弱了帧间时序信息；
- ILGP 使用原始 RGB 帧序列，VLM 能从帧间差异中读取运动轨迹；
- **NavLogic 的消融（N=1 vs N=10）必须给出正面结果**，否则帧历史设计失去支撑。

VLN-R1（2025）的 Long-Short Memory Sampling（近期帧 + 远期帧均衡采样）是一个有意思的改进方向：如果 NavLogic 的实验显示 N=10 效果有限，可以考虑引入不均匀采样策略作为改进点。

---

## 三、研究空白分析 / Gap Analysis

基于以上 30+ 篇文献，可以可视化各维度的覆盖情况：

```
                       无预建图 | 拓扑记忆 | 模糊/长指令 | 帧历史 | 失败纠错 | 真实机器人
MSNav (2025)              ✓   |    ✓    |     ✓      |  部分  |    ✗    |     ✗
GC-VLN CoRL'25            ✓   |    ✗    |     ✓      |   ✗   |    ✓    |     ✗
SmartWay IROS'25          ✓   |    ✗    |     ✗      |   ✓   |    ✓    |     ✓
EmergeNav (2026)          ✓   |    ✓    |     ✗      |  隐含  |    ✗    |     ✗
Mobility VLA (2024)       ✗   |    ✓    |     ✓      |   ✓   |    ✗    |     ✓
CA-Nav (2024)             ✓   |    ✗    |     ✓      |   ✓   |   部分  |     ✓
NaVid RSS'24              ✓   |    ✗    |     ✗      |   ✓   |    ✗    |    有限
LIFGIF (2024)             ✓   |    ✓    |     ✓      |   ✗   |    ✗    |     ✓
─────────────────────────────────────────────────────────────────────────────
NavLogic (目标)           ✓   |    ✓    |     ✓      |   ✓   |    ✓    |     ✓
```

**结论：目前没有任何一篇论文同时满足全部六列特征。NavLogic 的组合创新是真实存在的研究空白。**

最危险的竞争对手：
- **MSNav**（2025, arXiv）：最接近，缺失帧历史和失败纠错，且为仿真；需重点区分
- **SmartWay**（IROS 2025）：有真机+回退，但无拓扑记忆，无模糊指令处理
- **GC-VLN**（CoRL 2025）：有约束回退+模糊指令，但无拓扑记忆，无帧历史，无真机

---

## 四、对 NavLogic 设计的启示 / Design Implications

| 文献发现 | 对 NavLogic 的启示 |
|---|---|
| VLMnav 消融：K帧历史未改善性能 | 必须设计能让 VLM 利用帧间差异的提示词；考虑 Long-Short Memory Sampling (VLN-R1) |
| SmartWay：自适应回退在真实 Turtlebot 上有效 | 纠错机制在真实机器人上可行，验证了 NavLogic 路线；可参考其回退触发条件设计 |
| MSNav：动态拓扑地图 + 节点剪枝 | iLTM 需要类似的节点管理策略，防止地图无限增长 |
| CA-Nav：约束感知子指令完成追踪 | NavLogic 双粒度指令分层的良好工程参照；区别在于过程指令"可旁路"而非"必须完成" |
| EmergeNav 2026：双记忆零样本 VLN 达新 SOTA | 该论文（2026年3月）可能是 NavLogic 投稿时最强基线，需纳入对比 |
| GC-VLN：将指令形式化为约束图 | NavLogic 的指令分解可参考此形式化；同时明确区分：本文处理的是连续室内真实环境，GC-VLN 在仿真离散图上验证 |
| LIFGIF：factor-graph SLAM + 3D地标图 + Spot机器人 | 本文最相关的真实机器人拓扑图先行工作；NavLogic 的 iLTM 不依赖 SLAM，轻量更多 |
| Mind the Error! IROS'24：指令错误使 SR 下降 25% | 模糊指令的实际影响有实证，可引用作为 NavLogic 动机数据 |

---

## 五、建议引用的核心文献（NavLogic 论文 Related Work 必引）

### 必引（直接竞争/最相关）
1. **MSNav** (2508.16654) — 最近似竞争者，必须详细区分
2. **NaVid** (2402.15852, RSS 2024) — ILGP 帧历史设计的最强先行工作
3. **SmartWay** (2503.10069, IROS 2025) — 真实机器人回退纠错的最强先行工作
4. **GC-VLN** (2509.10454, CoRL 2025) — 指令图约束 + 回退，直接相关
5. **EmergeNav** (2603.16947, 2026) — 最新 SOTA 无地图 VLN，必须对比

### 强烈建议引用
6. **CA-Nav** (2412.10137) — 子指令分解 + 约束追踪，工程参照
7. **E²BA** (2311.00530, IEEE TCSVT 2025) — 模糊目标 + 回溯判别器
8. **LH-VLN** (2412.09082, CVPR 2025) — 长时序 VLN 基准和动机
9. **LIFGIF** (2411.07848) — 真实机器人 + 在线3D拓扑图（Spot）
10. **CorrectNav** (2508.10416) — 自我纠错（作为对比方法）
11. **EventNav** (2408.02535, CIKM 2024) — 事件知识图 + 动态历史回溯

### 背景引用（Introduction 动机段）
12. **Mind the Error!** (2403.10700, IROS 2024) — 模糊指令使 SR 下降 25%
13. **VLMaps** (2210.05714, ICRA 2023) — 语义地图导航（对比无图优势）
14. **LM-Nav** (2207.04429, CoRL 2022) — 拓扑图导航（预建图对比）
15. **NavGPT** (2305.16986, AAAI 2024) — LLM历史推理导航

---

## 六、文献数量统计

| 主题 | 检索到论文数 | 直接相关 |
|---|---|---|
| 无地图 VLN + 拓扑/场景记忆 | 13 | 8 |
| 帧历史 / 时序上下文 | 8 | 6 |
| 失败恢复与回退 | 8 | 5 |
| 模糊/长指令处理 | 8 | 5 |
| **总计（去重后）** | **~30** | **~20** |

---

*文献检索基于三路并行 arXiv + Web 搜索，覆盖 2022–2026 年，重点关注 2024–2026 年最新工作。*
