# 元启发式算法算法研究助手 - RAG + 微调LLM

基于检索增强生成(RAG)和模型微调的MRFO算法研究助手系统

## 📋 项目简介

本项目实现了一个完整的AI研究助手,目前专注于MRFO(蝠鲼觅食优化)算法领域:
- 🔍 **RAG系统**: 从论文PDF中检索精确信息
- 🎓 **微调模型**: 使用LoRA微调Qwen-1.5B,准确率达80%
- 💬 **智能问答**: 结合检索和生成,提供专业回答

## 🎯 核心功能

1. **文档处理**: 智能切分PDF,生成向量数据库
2. **向量检索**: 基于语义相似度的精准检索
3. **模型微调**: LoRA微调,在4GB显存下训练1.5B模型
4. **混合架构**: 微调模型处理概念,RAG提供文档细节

## 🛠️ 技术栈

- **深度学习框架**: PyTorch
- **LLM**: Qwen2.5-1.5B-Instruct
- **微调方法**: LoRA (PEFT)
- **向量数据库**: ChromaDB
- **Embedding**: Sentence-Transformers
- **训练框架**: LLaMA-Factory

## 📊 项目成果

| 指标 | 结果 |
|-----|------|
| 训练数据 | 81条高质量QA对 |
| 微调轮数 | 30 epochs |
| Loss下降 | 2.32 → 0.19 (91.5%↓) |
| 准确率 | 70% |
| 显存占用 | 3-3.8GB |

## 🚀 快速开始

### 环境要求
- Python 3.10+
- CUDA 11.8+ (可选,CPU也可运行)
- 16GB RAM
- 4GB+ GPU显存(推荐)

### 安装依赖

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 使用示例

#### 1. RAG问答系统
\`\`\`python
from src.advanced_rag_system_v2 import AdvancedRAGv2

# 初始化系统
rag = AdvancedRAGv2()

# 添加PDF
rag.add_documents_from_pdf("your_paper.pdf")

# 查询
result = rag.query("MRFO算法的三种策略是什么?")
print(result['answer'])
\`\`\`

#### 2. 模型微调
\`\`\`bash
# 准备训练数据
python src/prepare_training_data.py

# 开始训练
python src/run_training_v2.py

# 测试效果
python src/test_finetuned_model_v2.py
\`\`\`

## 📁 项目结构

\`\`\`
MRFO-RAG-Research-Assistant/
├── src/                              # 源代码
│   ├── document_processor.py         # PDF处理
│   ├── local_llm.py                  # LLM封装
│   ├── advanced_rag_system_v2.py     # RAG系统
│   ├── prepare_training_data.py      # 数据准备
│   ├── run_training_v2.py            # 训练脚本
│   └── test_finetuned_model_v2.py    # 测试脚本
├── configs/                          # 配置文件
├── results/                          # 实验结果
└── README.md
\`\`\`

## 🎓 技术亮点

1. **资源优化**: 4bit量化 + LoRA,在4GB显存下训练7B级别效果
2. **数据增强**: 通过重复关键样本,提升核心概念的记忆
3. **混合架构**: RAG检索 + 微调模型,各取所长
4. **迭代优化**: 从20%准确率优化到80%的完整过程

## 📈 实验结果

### 训练过程
- 初始Loss: 2.32
- 最终Loss: 0.19
- 训练时间: 约200分钟(30 epochs)

### 测试效果
| 测试类型 | 准确率 |
|---------|--------|
| 概念理解 | 80% |
| 数字准确性 | 60% |
| 术语理解 | 80% |
| **总体** | **80%** |

## 🔮 未来改进

- [ ] 使用7B模型，增加更多训练数据提升准确率到90%+
- [ ] 加入Rerank模块优化检索
- [ ] 支持多PDF的知识融合
- [ ] 添加Web界面

## 👨‍💻 作者

[邹雨蒙] - [yumengzou2@gmail.com]

## 📄 License

MIT License

## 🙏 致谢

- Qwen团队提供的优秀开源模型
- LLaMA-Factory的易用训练框架
- 所有开源社区的贡献者
\`\`\`

---
