# llm_with_rag.py
"""
极简 RAG 查询脚本：
  - 加载 chroma_db
  - 对 user query 做检索（top_k）
  - 把检索到的片段拼成 prompt
  - 通过 OpenAI-compatible client (你提供的 qwen 接口) 调用模型生成回答
用法：
  export QWEN_API_KEY="..."
  python llm_with_rag.py --persist_dir chroma_db --k 3
"""

import os
import argparse
import logging
from openai import OpenAI   # 用你之前的 only_llm 风格接入 qwen (openai-compatible)
from langchain_community.vectorstores import Chroma

# 添加QwenEmbeddings导入
from store_manager import QwenEmbeddings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_prompt(query: str, docs):
    parts = []
    for i, d in enumerate(docs, start=1):
        text = getattr(d, "page_content", getattr(d, "content", str(d)))
        src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
        parts.append(f"[片段 {i} | 来源: {src}]\n{text}\n")
    context = "\n".join(parts)
    prompt = (
        "下面是检索到的知识片段（仅供参考）。请**仅基于这些片段**回答用户问题，"
        "如果片段中没有相关信息，请回答“我不知道”。\n\n"
        f"{context}\n用户问题：{query}\n\n请给出简洁准确的回答，并在末尾列出引用来源。"
    )
    return prompt

def qwen_chat(prompt, model="qwen3-max", temperature=0.1):
    client = OpenAI(api_key=os.environ.get("QWEN_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that must only answer based on provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    return completion.choices[0].message.content

def main():
    import os
    from pathlib import Path
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="RAG问答测试")
    parser.add_argument("--persist_dir", "-p", default="chroma_db", help="向量数据库路径")
    parser.add_argument("--k", type=int, default=3, help="检索top-k数量")
    parser.add_argument("--query", "-q", default="如何更换电池?", help="查询问题")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        print("❌ 错误：请设置环境变量 QWEN_API_KEY")
        exit(1)
    
    print("=" * 70)
    print("🔍 RAG问答测试")
    print("=" * 70)
    
    # 检查向量数据库
    if not os.path.exists(args.persist_dir):
        print(f"❌ 错误：向量数据库不存在: {args.persist_dir}")
        print("💡 提示：请先运行 store_manager.py 构建向量数据库")
        exit(1)
    
    # 加载向量数据库
    print(f"\n📂 正在加载向量数据库...")
    print(f"   数据库路径: {args.persist_dir}")
    
    emb = QwenEmbeddings()
    vect = Chroma(persist_directory=args.persist_dir, embedding_function=emb)
    retriever = vect.as_retriever(search_kwargs={"k": args.k})
    
    # 获取数据库统计信息
    try:
        count = vect._collection.count() if hasattr(vect, '_collection') else "未知"
        print(f"✅ 加载成功")
        print(f"   - 向量数量: {count}")
        print(f"   - 检索top-k: {args.k}")
    except:
        print(f"✅ 加载成功")
    
    # 执行检索
    query = args.query
    print(f"\n❓ 用户问题: {query}")
    print(f"\n🔍 正在检索相关文档...")
    
    start_time = datetime.now()
    docs = retriever.invoke(query)
    retrieval_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ 检索完成！")
    print(f"   - 检索耗时: {retrieval_time:.3f} 秒")
    print(f"   - 检索到片段: {len(docs)} 个")
    
    if not docs:
        print("⚠️  警告：未检索到相关文档")
        exit(1)
    
    # 显示检索到的片段
    print(f"\n📄 检索到的文档片段:")
    print("-" * 70)
    for i, d in enumerate(docs, start=1):
        content = getattr(d, "page_content", getattr(d, "content", str(d)))
        src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
        print(f"\n片段 {i} (来源: {src}):")
        print(f"  {content[:150]}..." if len(content) > 150 else f"  {content}")
    print("-" * 70)
    
    # 构建Prompt
    print(f"\n📝 正在构建Prompt...")
    prompt = make_prompt(query, docs)
    print(f"✅ Prompt构建完成")
    print(f"   - Prompt长度: {len(prompt)} 字符")
    print(f"\n📋 Prompt预览（前300字符）:")
    print("-" * 70)
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    print("-" * 70)
    
    # 调用LLM生成回答
    print(f"\n🤖 正在调用LLM生成回答...")
    print(f"   - 模型: qwen3-max")
    print(f"   - Temperature: 0.1 (低温度，保证准确性)")
    
    start_time = datetime.now()
    answer = qwen_chat(prompt, model="qwen3-max", temperature=0.1)
    generation_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ 回答生成完成！")
    print(f"   - 生成耗时: {generation_time:.2f} 秒")
    
    # 输出结果
    print(f"\n{'='*70}")
    print("➡️  RAG回答:")
    print(f"{'='*70}")
    print(answer)
    print(f"{'='*70}")
    
    # 输出引用来源
    print(f"\n📚 引用来源:")
    print("-" * 70)
    sources_info = []
    for i, d in enumerate(docs, start=1):
        content = getattr(d, "page_content", getattr(d, "content", str(d)))
        src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
        print(f"{i}. {src}")
        print(f"   预览: {content[:80]}...")
        sources_info.append({"index": i, "source": src, "content": content[:100]})
    
    # 保存结果
    output_file = output_dir / "rag_answer.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RAG问答结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"问题: {query}\n\n")
        f.write("回答:\n")
        f.write(answer)
        f.write("\n\n")
        f.write("引用来源:\n")
        for info in sources_info:
            f.write(f"{info['index']}. {info['source']}\n")
            f.write(f"   {info['content']}...\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(f"检索耗时: {retrieval_time:.3f} 秒\n")
        f.write(f"生成耗时: {generation_time:.2f} 秒\n")
        f.write(f"总耗时: {retrieval_time + generation_time:.2f} 秒\n")
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"\n💡 提示：")
    print(f"   - 检索耗时: {retrieval_time:.3f} 秒")
    print(f"   - 生成耗时: {generation_time:.2f} 秒")
    print(f"   - 总耗时: {retrieval_time + generation_time:.2f} 秒")

if __name__ == "__main__":
    main()