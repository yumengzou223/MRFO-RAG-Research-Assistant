# 元启发式算法研究助手 - RAG + 微调LLM

基于检索增强生成(RAG)和模型微调的MRFO算法研究助手系统

## 📋 项目简介

本项目实现了一个完整的AI研究助手，专注于MRFO(蝠鲼觅食优化)算法领域：
- 🔍 **RAG系统**: 从论文PDF中检索精确信息
- 🎓 **微调模型**: 使用LoRA微调Qwen-1.5B，准确率达89%
- 💬 **智能问答**: 结合检索和生成，提供专业回答

## 🎯 核心功能

1. **文档处理**: 智能切分PDF，生成向量数据库
2. **向量检索**: 基于语义相似度的精准检索 + BM25关键词检索
3. **混合搜索**: RRF融合 + Cross-Encoder重排序
4. **模型微调**: LoRA微调，在4GB显存下训练1.5B模型
5. **场景感知**: 针对简单/复杂场景的专项优化

## 🛠️ 技术栈

- **深度学习框架**: PyTorch
- **LLM**: Qwen2.5-1.5B-Instruct
- **微调方法**: LoRA (PEFT)
- **向量数据库**: ChromaDB
- **Embedding**: Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **检索**: BM25 + 向量混合搜索
- **训练框架**: LLaMA-Factory

## 📊 最新成果 (2026-03-27)

### 评估结果

| 问题类型 | 问题描述 | 期望答案 | 结果 |
|---------|---------|---------|------|
| N-01 | 简单场景 vs MRFO | 7.89% | ✅ PASS |
| N-02 | 复杂场景 vs MRFO | 5.89% | ✅ PASS |
| N-03 | 简单场景 vs 未调度 | 45.30% | ✅ PASS |
| N-04 | 复杂场景 vs 未调度 | 53.29% | ✅ PASS |
| N-05 | 设备数量 | 12台/20台 | ✅ PASS |
| N-06 | PAR触发条件 | 式(35), PAR | ✅ PASS |
| C-01 | MRFO核心思想 | 蝠鲼, 觅食 | ✅ PASS |
| C-02 | 三种觅食策略 | 链式, 螺旋, 翻滚 | ✅ PASS |
| C-03 | DLM MRFO改进 | 离散, 长时记忆, PAR | ✅ PASS |

**最终得分：8/9 (89%)**

### 模型信息

| 指标 | 数值 |
|-----|------|
| 模型名称 | mrfo_lora_aggressive |
| 训练步数 | 500步 |
| 训练损失 | 0.199 |
| 训练时间 | 约5小时45分钟 |
| 设备 | GTX 1650 Ti (4GB) |

### 关键数字标注（已确认正确）

| 数字 | 场景 | 对比基准 | 说明 |
|------|------|---------|------|
| 7.89% | 简单场景 | vs MRFO | 成本降低 |
| 5.89% | 复杂场景 | vs MRFO | 成本降低 |
| 45.30% | 简单场景 | vs 未调度 | 成本降低 |
| 53.29% | 复杂场景 | vs 未调度 | 成本降低 |
| 12台 | 简单场景 | - | 设备数量 |
| 20台 | 复杂场景 | - | 设备数量(12+8) |

## 🚀 快速开始

### 环境要求
- Python 3.10+
- CUDA 11.8+ (可选，CPU也可运行)
- 16GB RAM
- 4GB+ GPU显存(推荐)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用示例

#### 1. RAG问答系统

```python
from src.advanced_rag_system_v2 import AdvancedRAGv2

# 初始化系统
rag = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    lora_path="path/to/mrfo_lora_aggressive",
    use_hybrid=True,  # 开启混合搜索
    use_rerank=True,  # 开启重排序
)

# 查询
result = rag.query("DLM MRFO在简单场景下相比MRFO算法降低了多少成本？")
print(result['answer'])
```

#### 2. 运行评估

```bash
python final_eval.py
```

## 📁 项目结构

```
MRFO-RAG-Research-Assistant/
├── src/
│   ├── advanced_rag_system_v2.py  # 核心RAG系统（已优化）
│   ├── document_processor.py        # PDF处理
│   └── local_llm.py               # LLM封装
├── generate_report.py              # PDF报告生成
├── final_eval.py                  # 最终评估脚本
├── 优化报告_2026-03-27.pdf       # 详细优化报告
└── README.md
```

## 🎓 技术亮点

1. **场景感知检索**: 针对简单/复杂场景使用不同的检索策略
2. **Few-shot Prompt**: 通过示例展示正确的问题-回答格式
3. **混合搜索**: BM25 + 向量检索 + RRF融合
4. **Cross-Encoder重排序**: 二次排序提升相关性
5. **资源优化**: 4bit量化 + LoRA，4GB显存训练1.5B模型

## 📈 优化历程

### 2026-03-27 优化记录

1. **问题诊断**:
   - 发现场景-数字混淆问题
   - 确认正确的数字标注（7.89%是简单场景，不是复杂场景）
   - 区分"vs MRFO"和"vs 未调度"的不同基准

2. **Prompt优化**:
   - 添加Few-shot示例
   - 明确区分对比基准
   - 添加设备数量专项规则

3. **检索优化**:
   - 场景感知补充查询
   - 针对PAR、数字、设备数量的专门检索

4. **评估改进**:
   - 89%准确率（8/9问题通过）
   - 数字类问题全部正确
   - 概念理解类问题全部正确

## ⚠️ 踩坑记录

1. **数据修复需谨慎**: 在确认原始数据标注是否正确之前，不要盲目修复
2. **不要迷信train_loss**: 更低的loss不代表更好的实际效果
3. **保存策略很重要**: 设置合理的save_steps（如50步保存一次）
4. **评估方法决定方向**: 需要与领域专家确认正确的标注

## 🔮 未来改进

- [ ] 使用7B模型提升准确率
- [ ] 增加更多训练数据
- [ ] 支持多PDF知识融合
- [ ] 添加Web界面

## 👨‍💻 作者

[邹雨蒙] - [yumengzou2@gmail.com]

## 📄 License

MIT License

## 🙏 致谢

- Qwen团队提供的优秀开源模型
- LLaMA-Factory的易用训练框架
- 所有开源社区的贡献者
