"""
MRFO研究助手 - Gradio界面
===================================

## 说明

这个脚本创建了一个简单的网页界面，让用户可以和MRFO RAG系统对话。

### 依赖安装
pip install gradio

### 运行方法
python gradio_demo.py
然后打开浏览器访问 http://localhost:7860

===================================
"""
import os
import sys

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 添加项目路径
WORK_DIR = r'D:\Anaconda\envs\rag_project\MRFO-RAG-Research-Assistant'
PARENT_DIR = r'D:\Anaconda\envs\rag_project'
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, WORK_DIR)

import gradio as gr
from src.advanced_rag_system_v2 import AdvancedRAGv2

print("=" * 60)
print("MRFO研究助手 - Gradio界面")
print("=" * 60)

# ============================================================
# 第1步：初始化RAG系统
# ============================================================
print("\n[1/3] 正在初始化RAG系统...")

# 注意：LoRA模型(mrfo_lora_aggressive)训练数据偏差，问MRFO也答成DLM MRFO
# 暂时使用基础模型，待重新训练正确的LoRA
rag = AdvancedRAGv2(
    collection_name="research_knowledge_base_v5",
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    lora_path=None,  # 暂不使用LoRA
    use_hybrid=True,   # 开启混合搜索(BM25+向量)
    use_rerank=True,   # 开启重排序
    vector_top_k=20,   # 向量检索返回20条
    bm25_top_k=20,     # BM25检索返回20条
    final_top_k=8,     # 最终使用8条
    rerank_top_k=12    # 重排序候选12条
)
print("✅ RAG系统初始化完成!")

# ============================================================
# 第2步：定义问答函数
# ============================================================
def chat_with_rag(message, history):
    """
    这是核心的问答函数
    - message: 用户当前输入的问题
    - history: 对话历史记录
    返回值会显示在界面上
    """
    print(f"\n收到问题: {message}")
    
    try:
        # 调用RAG系统获取答案
        result = rag.query(message, show_sources=False)
        answer = result['answer']
        
        # 在控制台打印，方便调试
        print(f"回答: {answer[:100]}...")
        
        # 返回给Gradio显示
        return answer
        
    except Exception as e:
        error_msg = f"出错了: {str(e)}"
        print(error_msg)
        return error_msg

# ============================================================
# 第3步：创建Gradio界面
# ============================================================
print("\n[2/3] 正在创建界面...")

demo = gr.ChatInterface(
    fn=chat_with_rag,
    title="🔍 MRFO研究助手",
    description="基于RAG的MRFO算法智能问答系统<br>"
                "可以问：MRFO算法原理、DLM MRFO改进、数字指标等问题",
    
    textbox=gr.Textbox(
        label="输入问题",
        placeholder="例如：MRFO算法的三种觅食策略是什么？",
        lines=2
    ),
    
    theme="soft",
    
    examples=[
        ["MRFO算法的核心思想是什么？"],
        ["DLM MRFO在简单场景下相比MRFO降低了多少成本？"],
        ["MRFO算法的三种觅食策略是什么？"],
        ["PAR专项优化策略是什么？"],
        ["简单场景和复杂场景分别有多少台设备？"],
    ]
)

print("✅ 界面创建完成!")

# ============================================================
# 第4步：启动服务
# ============================================================
print("\n[3/3] 启动服务...")
print("=" * 60)
print("🎉 访问地址: http://localhost:7860")
print("📝 按 Ctrl+C 停止服务")
print("=" * 60)

demo.launch(
    server_name="0.0.0.0",
    share=False,
    inbrowser=True
)

print("\n服务已关闭")
