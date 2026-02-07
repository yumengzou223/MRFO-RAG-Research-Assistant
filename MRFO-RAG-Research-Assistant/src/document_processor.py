"""
文档处理模块: 从PDF提取文本并智能切分
"""
import re
from typing import List, Dict
import PyPDF2


class DocumentProcessor:
    def __init__(self, chunk_size=500, overlap=50):
        """
        Args:
            chunk_size: 每个chunk的目标字符数
            overlap: chunk之间的重叠字符数(保证上下文连贯)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, pdf_path: str) -> str:
        """
        从PDF提取文本

        Args:
            pdf_path: PDF文件路径
        Returns:
            提取的全部文本
        """
        print(f"📄 正在读取PDF: {pdf_path}")

        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                print(f"📖 总页数: {total_pages}")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += page_text
                    print(f"  ✓ 已处理 {page_num}/{total_pages} 页")

            print(f"✅ PDF读取完成,共 {len(text)} 字符\n")
            return text

        except Exception as e:
            print(f"❌ 读取PDF失败: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        清理文本(去除多余空格、换行等)
        """
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊字符(根据需要调整)
        text = text.strip()
        return text

    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        按句子切分文本(智能方法)

        策略:
        1. 先按句号、问号、感叹号切分成句子
        2. 把句子组合成chunk,保持在chunk_size左右
        3. chunk之间有overlap,保证上下文连贯
        """
        print(f"🔪 开始切分文本...")
        print(f"   目标chunk大小: {self.chunk_size} 字符")
        print(f"   重叠区域: {self.overlap} 字符\n")

        # 按句子切分(中英文标点)
        sentences = re.split(r'([。!?\.!\?])', text)
        # 把标点符号附加回去
        sentences_with_punct = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentences_with_punct.append(sentences[i] + sentences[i + 1])

        # 组合成chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences_with_punct:
            # 如果加上这句话会超过chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:  # 保存当前chunk
                    chunks.append(current_chunk.strip())
                    # 保留overlap部分作为下一个chunk的开头
                    current_chunk = current_chunk[-self.overlap:] + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += sentence

        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        print(f"✅ 切分完成! 共 {len(chunks)} 个chunks")
        print(f"   平均chunk长度: {sum(len(c) for c in chunks) // len(chunks)} 字符\n")

        return chunks

    def chunk_by_fixed_size(self, text: str) -> List[str]:
        """
        按固定大小切分(简单方法)
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap  # 后退overlap,保证重叠

        return chunks

    def process_pdf(self, pdf_path: str, method='sentences') -> List[Dict]:
        """
        完整处理流程: PDF → chunks

        Args:
            pdf_path: PDF路径
            method: 'sentences' 或 'fixed'
        Returns:
            List of {text: str, metadata: dict}
        """
        # 1. 读取PDF
        raw_text = self.load_pdf(pdf_path)

        if not raw_text:
            return []

        # 2. 清理文本
        cleaned_text = self.clean_text(raw_text)

        # 3. 切分
        if method == 'sentences':
            chunks = self.chunk_by_sentences(cleaned_text)
        else:
            chunks = self.chunk_by_fixed_size(cleaned_text)

        # 4. 添加metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk,
                'metadata': {
                    'source': pdf_path,
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
            })

        return processed_chunks


# ========== 测试代码 ==========
def demo():
    """
    演示文档处理功能
    """
    print("=" * 60)
    print("📚 文档处理器演示")
    print("=" * 60)
    print()

    # 创建处理器
    processor = DocumentProcessor(chunk_size=300, overlap=50)

    # 测试1: 处理示例文本
    print("🧪 测试1: 处理示例文本")
    print("-" * 60)

    sample_text = """
    大语言模型(Large Language Model, LLM)是一种基于深度学习的自然语言处理模型。
    它通过在海量文本数据上进行预训练,学习到丰富的语言知识和推理能力。
    目前主流的LLM包括GPT系列、BERT、LLaMA等。
    这些模型在问答、翻译、代码生成等任务上表现出色。

    RAG(Retrieval-Augmented Generation)是一种结合检索和生成的技术。
    它通过检索相关文档,为LLM提供额外的上下文信息。
    这样可以减少模型的幻觉问题,提高回答的准确性。
    RAG系统通常包括文档处理、向量检索、提示工程等模块。
    """

    chunks = processor.chunk_by_sentences(sample_text)

    print("📋 切分结果:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  长度: {len(chunk)} 字符")
        print(f"  内容: {chunk[:100]}...")

    # 测试2: 如果你有PDF文件
    print("\n" + "=" * 60)
    print("🧪 测试2: 处理PDF文件(可选)")
    print("-" * 60)
    print("提示: 把你的PDF文件放在项目文件夹,然后取消下面的注释\n")

    # 取消注释来测试PDF处理:
    pdf_path = "3.3.pdf"
    chunks = processor.process_pdf(pdf_path, method='sentences')

    print(f"✅ 处理完成!")
    print(f"   总chunks数: {len(chunks)}")
    print(f"\n前3个chunks:")
    for chunk_data in chunks[:3]:
      print(f"\n  Chunk {chunk_data['metadata']['chunk_id']}:")
      print(f"    {chunk_data['text'][:150]}...")


if __name__ == "__main__":
    demo()