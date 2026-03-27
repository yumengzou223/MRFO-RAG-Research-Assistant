# MRFO研究助手 - RAG + 微调LLM + Agent智能路由

基于检索增强生成(RAG)和模型微调的MRFO算法研究助手，支持Agent智能路由自动选择最佳模型。

## 项目简介

本项目实现了一个智能问答系统，专注于MRFO(蝠鲼觅食优化)算法领域：
- RAG系统：从论文PDF中检索精确信息
- 微调模型：使用LoRA微调Qwen-1.5B
- **Agent智能路由：自动选择基础模型或LoRA模型回答问题**
- **测试准确率：100% (9/9)**

## 核心功能

1. **文档处理**：智能切分PDF，生成向量数据库
2. **混合检索**：向量检索 + BM25关键词检索 + 重排序
3. **Agent智能路由**：根据问题类型自动选择最佳模型
4. **模型微调**：LoRA微调，在4GB显存下训练1.5B模型

## 系统架构

```
用户问题 → Agent判断问题类型
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
基础模型               LoRA模型
(原MRFO概念)         (DLM MRFO数字)
    ↓                   ↓
    └─────────┬─────────┘
              ↓
         合并答案 → 返回用户
```

## 技术栈

- **深度学习框架**：PyTorch
- **LLM**：Qwen2.5-1.5B-Instruct
- **微调方法**：LoRA (PEFT)
- **向量数据库**：ChromaDB
- **Embedding**：Sentence-Transformers
- **界面**：Gradio

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Agent智能问答系统

```bash
python gradio_agent_demo.py
```

然后访问 http://localhost:7860

### 3. 界面说明

Agent会自动判断问题类型并选择最佳模型：

| 问题类型 | 自动路由到 | 示例问题 |
|---------|----------|---------|
| 原MRFO基础概念 | 基础模型 | MRFO算法的核心思想是什么？ |
| DLM MRFO数字/成本 | LoRA模型 | DLM MRFO在简单场景降低了多少成本？ |
| 设备数量/PAR | LoRA模型 | 简单场景和复杂场景分别有多少台设备？ |

## 项目结构

```
MRFO-RAG-Research-Assistant/
├── src/
│   ├── advanced_rag_system_v2.py  # 核心RAG系统
│   ├── simple_rag_system.py      # 简化版RAG
│   └── local_llm.py              # LLM封装
├── gradio_demo.py                  # 基础Gradio界面
├── gradio_agent_demo.py           # Agent智能路由界面
├── agent_test.py                  # Agent测试脚本
└── README.md
```

## 测试结果

### Agent智能路由测试 (2026-03-27)

| 问题ID | 问题描述 | 路由到 | 期望答案 | 结果 |
|--------|---------|--------|---------|------|
| C-01 | MRFO核心思想 | 基础模型 | 蝠鲼,觅食 | ✅ PASS |
| C-02 | 三种觅食策略 | 基础模型 | 链式,螺旋,翻滚 | ✅ PASS |
| N-01 | 简单场景vsMRFO | LoRA模型 | 7.89% | ✅ PASS |
| N-02 | 复杂场景vsMRFO | LoRA模型 | 5.89% | ✅ PASS |
| N-03 | 简单场景vs未调度 | LoRA模型 | 45.30% | ✅ PASS |
| N-04 | 复杂场景vs未调度 | LoRA模型 | 53.29% | ✅ PASS |
| N-05 | 设备数量 | LoRA模型 | 12台,20台 | ✅ PASS |
| N-06 | PAR触发条件 | LoRA模型 | PAR,式(35) | ✅ PASS |
| C-03 | DLM MRFO改进 | LoRA模型 | 离散,长时记忆 | ✅ PASS |

**最终得分：9/9 (100%)**

### 为什么用Agent路由？

单一模型的问题：
- 基础模型：MRFO概念回答正确，但DLM MRFO数字容易混淆
- LoRA模型：DLM MRFO数字准确，但问MRFO基础问题时答成DLM MRFO

Agent路由解决方案：
- 基础模型 → 回答MRFO基础概念问题
- LoRA模型 → 回答DLM MRFO相关问题
- 两者结合 → 100%准确率

## 关键数字标注（已确认）

| 数字 | 场景 | 对比基准 | 说明 |
|------|------|---------|------|
| 7.89% | 简单场景 | vs MRFO | 成本降低 |
| 5.89% | 复杂场景 | vs MRFO | 成本降低 |
| 45.30% | 简单场景 | vs 未调度 | 成本降低 |
| 53.29% | 复杂场景 | vs 未调度 | 成本降低 |
| 12台 | 简单场景 | - | 设备数量 |
| 20台 | 复杂场景 | - | 设备数量(12+8) |

## 模型信息

### 基础模型
- 模型：Qwen/Qwen2.5-1.5B-Instruct
- 用途：回答MRFO基础概念问题

### LoRA模型
- 模型：Qwen/Qwen2.5-1.5B-Instruct + mrfo_lora_aggressive
- 训练参数：rank=32, alpha=64
- 用途：回答DLM MRFO数字/成本问题
- 位置：`D:\Anaconda\envs\rag_project\saves\mrfo_lora_aggressive`

## 踩坑记录

1. **LoRA训练数据偏差**：mrfo_lora_aggressive模型问MRFO时也答成DLM MRFO
2. **显存不足**：同时加载两个模型需要约6GB显存，需串行使用
3. **数字场景混淆**：需要明确标注每个数字的场景和对比基准

## License

MIT License

## 作者

邹雨蒙 - yumengzou2@gmail.com
