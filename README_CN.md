<p align="left">
    <a href="README.md">English</a>&nbsp;｜&nbsp;中文
</p>
<br>

<p align="center">
 <img src="assets/logo-zh.png" alt="腾讯混元" width="400"/> <br>
</p>

<h1 align="center">ExplorationBench：在可验证的异星世界中衡量AI系统的探索能力</h1>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](#-许可证)
&nbsp;&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2609.30199-b31b1b)](https://arxiv.org/abs/2609.30199)
&nbsp;&nbsp;
[![Leaderboard](https://img.shields.io/badge/Leaderboard-10%20frontier%20systems-ffc107)](https://explorationbench.com/#leaderboard)
&nbsp;&nbsp;
[![Blog](https://img.shields.io/badge/Blog-EN%20%7C%20%E4%B8%AD%E6%96%87-624aff)](https://explorationbench.com/blog/)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab)](#-快速开始)

</div>

<p align="center">
    🖥️&nbsp;<a href="https://explorationbench.com"><b>官方网站</b></a>&nbsp;&nbsp;|&nbsp;&nbsp;
    📮&nbsp;<a href="#-评测你的模型"><b>评测你的模型</b></a></p>

---

## 🔥 新闻

- **2026.09**：🎉 ExplorationBench发布：两个可验证的异星世界，共55个发现目标和140道留出题，并给出10个前沿AI系统的评测结果。[[论文]](https://arxiv.org/abs/2609.30199) [[网站]](https://explorationbench.com) [[博客]](https://explorationbench.com/blog/)

## 📋 目录

- [概览](#-概览)
- [排行榜](#-排行榜)
- [主要发现](#-主要发现)
- [开源内容](#-开源内容)
- [快速开始](#-快速开始)
- [仓库结构](#-仓库结构)
- [评测你的模型](#-评测你的模型)
- [相关基准](#-相关基准)
- [许可证](#-许可证)
- [引用](#-引用)
- [联系我们](#-联系我们)

## 📄 概览

科学发现始于已知问题的尽头。在那里，AI系统必须自己探索：提出假设、设计实验，并根据结果不断迭代。ExplorationBench在**可验证的异星世界**（verifiable Alien Worlds）中衡量这种能力。这些世界的规则是可执行的，因此每个答案都能被精确检验；同时规则与熟悉的知识相冲突，因此仅靠回忆无法解题。

系统从一份有误的手册和一组固定示例出发，在四轮中通过唯一的工具探测这个世界，世界只返回确定性的反馈，不作任何解释。每轮结束后，我们复制一份禁用工具的会话，让它陈述自己认为成立的规则，并把每道留出题各答三遍。这份副本随后就被丢弃，因此测试题永远不会泄漏到探索过程中。

<p align="center">
  <img src="assets/overview.png" alt="ExplorationBench评测协议" width="90%">
</p>

*评测协议。每个系统都从同一份有误的手册和同一组固定示例出发，在四轮中自己选择探针。在每个里程碑M<sub>t</sub>，一份禁用工具的副本陈述它认为成立的规则，并解答留出题。*

<p align="center">
  <img src="assets/trajectories.png" alt="四轮探索中的留出题准确率" width="95%">
</p>

*自主探索过程中的留出题准确率。上：AlienCode；下：AlienLogic。每个小图突出一个系统从M<sub>0</sub>到M<sub>4</sub>的曲线，其余九个系统显示为灰色。每条曲线都是该系统的Best@3轨迹，每个里程碑的数值是每道留出题三次作答的平均。*

**概况**：2个沙盒，55个发现目标，140道留出题，10个前沿系统，每个系统在每个沙盒中跑3条独立轨迹；所有判定都来自解释器或证明检查器，无需LLM作为裁判。

**沙盒**：
- **AlienCode**：在一门小型计算语言中做程序合成。它的运算符看起来很熟悉，语义却是隐藏的：`SHATTER`执行的是乘法，`EMIT(100)`打印出`127`，而且没有任何原语能做加法。共31个发现目标和70道留出题；每个程序都由解释器在五组私有输入上判分。
- **AlienLogic**：在一阶自然演绎系统中做形式化证明，其中24条推理规则被修改过。共70道留出定理，其中25道不可证；每个证明都由证明检查器验证，不可证定理只有在系统拒绝证明时才得分。

## 🏆 排行榜

四轮自主探索之后（M<sub>4</sub>）的留出题准确率（%）。**Best@3**是三条独立轨迹中的最好成绩，**Mean@3**是三条轨迹的均值；M<sub>0</sub>是Best@3那条轨迹只读完固定示例时的准确率。每个分数都是每道题三次作答的平均，每个系统都使用其最高推理设置。更多指标、对照条件和可交互的轨迹见[网站](https://explorationbench.com/#leaderboard)。

**AlienCode**

| 排名 | 系统 | M<sub>0</sub> | Best@3 | Mean@3 |
|:---:|---|---:|---:|---:|
| 1 | Claude Opus 5 | 3.8 | **89.0** | 73.5 |
| 2 | GPT-5.6 Sol | 11.0 | 88.1 | 79.5 |
| 3 | Gemini 3.8 Flash | 1.4 | 79.0 | 39.7 |
| 4 | Kimi K3 | 2.4 | 79.0 | 38.7 |
| 5 | Grok 4.6 | 4.8 | 69.0 | 53.2 |
| 6 | Qwen3.8-Max | 2.4 | 65.7 | 64.0 |
| 7 | Hy4 preview | 0.5 | 63.3 | 30.2 |
| 8 | DeepSeek-V4.1-Flash | 1.4 | 59.5 | 46.3 |
| 9 | Seed2.1 Pro | 2.4 | 40.0 | 24.8 |
| 10 | DeepSeek-V4-Pro | 1.9 | 12.9 | 7.3 |

**AlienLogic**

| 排名 | 系统 | M<sub>0</sub> | Best@3 | Mean@3 |
|:---:|---|---:|---:|---:|
| 1 | Grok 4.6 | 42.4 | **83.8** | 72.1 |
| 2 | Claude Opus 5 | 42.9 | 77.6 | 73.3 |
| 3 | Qwen3.8-Max | 43.8 | 76.2 | 72.7 |
| 4 | GPT-5.6 Sol | 51.0 | 75.2 | 73.0 |
| 5 | Hy4 preview | 33.8 | 74.8 | 67.8 |
| 6 | DeepSeek-V4-Pro | 32.9 | 73.8 | 65.1 |
| 7 | Kimi K3 | 44.8 | 72.9 | 67.9 |
| 8 | DeepSeek-V4.1-Flash | 50.0 | 72.4 | 59.2 |
| 9 | Seed2.1 Pro | 37.1 | 67.6 | 59.5 |
| 10 | Gemini 3.8 Flash | 47.6 | 58.1 | 57.6 |

## 💡 主要发现

- **任务所需的知识来自探索，而不是回忆或单纯的思考**。读完固定示例后，AlienCode中没有任何轨迹超过15.7%；四轮之后，最好的轨迹达到89.0%。同样的轮数如果没有环境反馈，准确率只有0.5–11.0%。
- **系统自己设计实验时，探索的效果最好**。把系统自己最好的探针回放给它，它拿到的证据完全相同；但在AlienCode中，10个系统里仍有9个在自主探索时表现更好，中位差为17.4个百分点。
- **两个世界卡在不同的地方**。在AlienLogic中，直接告知规则就能达到93–97%，高于每个系统的最好轨迹，所以瓶颈在于发现规则。在AlienCode中，10个系统里有7个的最好轨迹超过了一开始就告知规则的成绩，所以瓶颈在于运用规则。
- **知道规则不等于能正确运用规则**。即使最终的规则报告把某道题需要的规则全都说对了，这些题也只有73.4%能被解出。
- **单一分数掩盖了很多信息**。在相同预算下，同一个系统的几条轨迹最终可能相差几十个百分点（Kimi K3在AlienCode中为5.7–79.0%）；一个系统在一个世界中的排名，几乎无法预测它在另一个世界中的排名（Spearman相关系数为0.35）；而且进步是跃迁式的，后面的轮次还可能把它抹掉。

## 📦 开源内容

规则一旦公开，世界测到的就是回忆而不是探索，所以两个评测世界的规则和测试实例不公开。理解和运行这个基准所需的其余内容都在这里：

- **论文实验所用的评测代码**：评测框架、协议、计分、对照条件、每题三次作答、开卷作答、重新计分和审计。只有与具体世界相关的模块不同，`explorationbench/sandboxes/code/world_data.py`列出了每一处不同的取值。
- **评测引擎**：AlienCode解释器前端和AlienLogic证明检查器，规则已拆分到规则包中。加载私有规则后，它们对实验日志中全部7,497个证明和7,651个程序都给出与原来相同的判定。
- **两个公开的演示世界**：用同样的流程构建，使用相同的语言、手册、证明格式、协议和题型。它们的规则是全新的，与评测世界的任何一条规则都不同。
- `examples/`中有两个演示世界的**示例**，`analysis/`中有论文图表和主表背后的**分析脚本**。分析脚本读取的是私有的实验日志，所以它们在这里用来说明每个数字是怎么算出来的，而不能直接复现结果。

| | 演示世界（公开） | 评测世界（不公开） |
|---|---|---|
| AlienCode | 5个发现目标、4个示例、5道留出题 | 31个发现目标、70道留出题 |
| AlienLogic | 3条侧条件、5个示例、5道留出定理（1道不可证） | 24条修改过的规则、70道留出定理（25道不可证） |

如需在不公开的评测世界上评测模型，请见[评测你的模型](#-评测你的模型)。

## 🚀 快速开始

需要Python 3.11或更高版本。检查演示世界只用到标准库；调用模型需要`httpx`，分析脚本需要`numpy`和`matplotlib`。

### 检查演示世界（无需API密钥）

```bash
cd explorationbench
python3 -m sandboxes.code.build_world     # 重新构建并检查AlienCode演示任务集
python3 -m sandboxes.logic.episodes       # 检查AlienLogic演示回合
python3 -m sandboxes.logic.certify        # 认证其中的不可证定理
cd ..
python3 examples/render.py                # 重新生成examples/中的示例
```

### 在演示世界上评测模型

评测框架通过OpenRouter路由访问任何兼容OpenAI的chat-completions接口：

```bash
pip install httpx
cd explorationbench
export OPENROUTER_API_KEY=<your_api_key>
# 默认使用OpenRouter；如有需要，可以改为任何其他兼容OpenAI的接口：
# export OPENROUTER_CHAT_URL=https://api.example.org/v1/chat/completions

# AlienCode：四轮探索，然后在每个里程碑对每道题作答三次
EVAL_TOOL_MODE=1 ALIENCODE_PROTOCOL_V2=1 python3 -u sandboxes/code/run_eval.py \
    --model openrouter/<model> --short <name> --run-id demo_n1
python3 -u scripts/answer_variance.py --run demo_n1 --trials 3 --milestones 0,1,2,3,4

# AlienLogic
EVAL_TOOL_MODE=1 ALIENLOGIC_PROTOCOL_V2=1 python3 -u sandboxes/logic/run_eval.py \
    --model openrouter/<model> --model-short <name> --defer-tests --skip-pre --run-id demo_l1
python3 -u scripts/logic_answer_variance.py --run demo_l1 --trials 3
```

结果、对话记录和HTML回放都写在`logs/`下。演示世界的分数仅供开发使用，不能与排行榜比较。

凭据和运行设置也可以放在本地文件中：把`config/eval.local.example.toml`复制为`explorationbench/eval.local.toml`（已被git忽略），或者用`EVAL_LOCAL_CONFIG`指向其他路径。环境变量的优先级高于该文件。

## 📁 仓库结构

```
ExplorationBench/
├── explorationbench/        评测代码，已装入公开的演示世界
│   ├── sandboxes/code/      AlienCode：评测框架、协议、解释器前端、手册、
│   │                        演示规则包和任务集、世界构建
│   ├── sandboxes/logic/     AlienLogic：评测框架、协议、证明检查器、演示侧条件、
│   │                        演示回合、不可证性证书
│   ├── common/              模型客户端（路由、会话、工具调用、闭卷副本）和运行时
│   ├── frameworks/          框架赛道接口
│   └── scripts/             编排、对照条件、每题三次作答、
│                            开卷作答、重新计分、审计、导出、HTML回放
├── analysis/                论文图表和主表背后的脚本
├── examples/                演示世界的示例
├── config/                  模型接口和凭据的模板
└── assets/                  本README用到的图片
```

## 📮 评测你的模型

两个评测世界不公开。如果希望在评测世界上评测你的模型，请[联系我们](#-联系我们)，我们会按与排行榜相同的协议运行评测。

## 🧭 相关基准

ExplorationBench是我们关于“模型如何在测试时获得新能力”系列基准中的第三个：

- [EvaLearn](https://github.com/ByteDance-Seed/EvaLearn)（NeurIPS 2025）：在一系列问题中从经验中学习。
- [CL-bench](https://github.com/Tencent-Hunyuan/CL-bench)：从上下文提供的知识中学习。
- **ExplorationBench**：在不提供现成知识的世界中，通过探索掌握陌生的规则。

## 📜 许可证

本仓库代码采用[Apache-2.0许可证](LICENSE)。

## 📝 引用

```bibtex
@article{zhang2026explorationbench,
  title   = {ExplorationBench: Measuring AI Systems' Exploration
             in Verifiable Alien Worlds},
  author  = {Zhang, Ming and Xiang, Zhenghao and
             Gao, Peizhong and Shen, Yujiong and Wang, Yuhui and
             Yue, Zhonghan and Dou, Shihan and Yin, Zhangyue and
             Ye, Junjie and Liu, Shichun and Zheng, Weihuang and
             Chen, Jiahao and Chen, Jiayi and Liu, Hongzhang and
             Shao, Jiaqi and Gui, Tao and Zhang, Qi and
             Huang, Xuanjing and Zheng, Suncong and Pan, Maxm},
  journal = {arXiv preprint arXiv:2609.30199},
  year    = {2026}
}
```

## 📧 联系我们

如有问题或评测需求，欢迎提交issue，或发邮件给Ming Zhang（mingzhang23@m.fudan.edu.cn）或Maxm Pan（maxmpan@tencent.com）。

<div align="center">
<sub>ExplorationBench · 复旦大学 · 腾讯混元 · 清华大学</sub>
</div>
