# chunker.py
"""
简单的 Chunker：把纯文本切成定长片段并写入 chunks.json
用法示例：
    python chunker.py

输出：
  - chunks.json: 切分后的 [{ "id": int, "source": filename, "text": "..."}]
说明：
  - 强制以 UTF-8 读取文本，避免 GBK 解码错误（请确保文件为 UTF-8）
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
from extractor import extract_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # move start forward with overlap
        start = end - overlap
        if start < 0:
            start = 0
        if (start >= L) or (end == L):
            break
    return chunks

def langchain_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 实现文本分块
    
    Args:
        text (str): 需要分块的文本
        chunk_size (int): 每个块的最大长度
        overlap (int): 块之间的重叠长度
        
    Returns:
        List[str]: 分块后的文本列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    
    # 将文本转换为 Document 对象进行分割
    documents = [Document(page_content=text)]
    splitted_docs = splitter.split_documents(documents)
    
    # 提取分割后的文本内容
    chunks = [doc.page_content.strip() for doc in splitted_docs]
    # 过滤掉空字符串
    chunks = [chunk for chunk in chunks if chunk]
    
    return chunks

def main():
    import os
    from pathlib import Path
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 配置参数
    chunk_size = 500
    overlap = 50
    
    print("=" * 70)
    print("✂️  文本切片测试")
    print("=" * 70)
    
    # 读取提取的文本
    pdf_content_file = output_dir / "pdf_content.txt"
    if not pdf_content_file.exists():
        pdf_content_file = Path("pdf_content.txt")
    
    if not pdf_content_file.exists():
        print(f"❌ 错误：找不到提取的文本文件")
        print("💡 提示：请先运行 extractor.py 提取PDF文本")
        exit(1)
    
    print(f"\n📖 正在读取文本文件...")
    print(f"   文件路径: {pdf_content_file}")
    
    with open(pdf_content_file, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    print(f"✅ 文本读取成功")
    print(f"   - 文本长度: {len(txt)} 字符")
    print(f"   - 切片参数: chunk_size={chunk_size}, overlap={overlap}")
    
    # ========== 方法1：简单切片 ==========
    print(f"\n{'='*70}")
    print("方法1：简单切片（固定长度切分）")
    print(f"{'='*70}")
    print(f"📝 开始切片...")
    
    chunks_simple = simple_chunk_text(txt, chunk_size, overlap)
    
    print(f"✅ 切片完成！")
    print(f"   - 切片数量: {len(chunks_simple)} 个")
    print(f"   - 平均长度: {sum(len(c) for c in chunks_simple) / len(chunks_simple):.0f} 字符")
    
    # 保存结果
    out_list = []
    for i, c in enumerate(chunks_simple):
        out_list.append({"id": i, "source": "test_user_manual.pdf", "text": c})
    
    output_file = output_dir / "chunks_simple.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    
    print(f"   - 输出文件: {output_file}")
    
    # 显示示例片段
    if chunks_simple:
        print(f"\n📄 示例片段（第1个，前150字符）:")
        print("-" * 70)
        print(chunks_simple[0][:150] + "..." if len(chunks_simple[0]) > 150 else chunks_simple[0])
        print("-" * 70)
    
    # ========== 方法2：LangChain智能切片 ==========
    print(f"\n{'='*70}")
    print("方法2：LangChain智能切片（按分隔符切分）")
    print(f"{'='*70}")
    print(f"📝 开始切片...")
    
    chunks_langchain = langchain_chunk_text(txt, chunk_size, overlap)
    
    print(f"✅ 切片完成！")
    print(f"   - 切片数量: {len(chunks_langchain)} 个")
    print(f"   - 平均长度: {sum(len(c) for c in chunks_langchain) / len(chunks_langchain):.0f} 字符")
    
    # 保存结果
    out_list = []
    for i, c in enumerate(chunks_langchain):
        out_list.append({"id": i, "source": "test_user_manual.pdf", "text": c})
    
    output_file = output_dir / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    
    print(f"   - 输出文件: {output_file}")
    
    # 显示示例片段
    if chunks_langchain:
        print(f"\n📄 示例片段（第1个，前150字符）:")
        print("-" * 70)
        print(chunks_langchain[0][:150] + "..." if len(chunks_langchain[0]) > 150 else chunks_langchain[0])
        print("-" * 70)
    
    # ========== 对比总结 ==========
    print(f"\n{'='*70}")
    print("📊 切片方法对比")
    print(f"{'='*70}")
    print(f"{'方法':<30} {'切片数量':<15} {'平均长度':<15}")
    print("-" * 70)
    avg_simple = sum(len(c) for c in chunks_simple) / len(chunks_simple) if chunks_simple else 0
    avg_langchain = sum(len(c) for c in chunks_langchain) / len(chunks_langchain) if chunks_langchain else 0
    print(f"{'简单切片':<30} {len(chunks_simple):<15} {avg_simple:.0f}")
    print(f"{'LangChain切片':<30} {len(chunks_langchain):<15} {avg_langchain:.0f}")
    print(f"{'='*70}")
    
    print(f"\n💡 提示：")
    print(f"   - 简单切片：固定长度切分，可能切断语义")
    print(f"   - LangChain切片：按分隔符智能切分，保持语义完整性")
    print(f"   - 建议使用 LangChain切片（chunks.json）进行后续处理")

if __name__ == "__main__":
    main()
