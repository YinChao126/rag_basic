# rag_basic_qwen_simple.py
"""
极简 RAG 示例（演示：切片 -> 向量化 -> 存储 -> 召回 -> 拼上下文 -> 调用 QWEN）

本示例展示了RAG系统的核心流程：
1. 文档读取与切片
2. 向量化与存储
3. 检索与生成
"""

import os
import sys
from glob import glob
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------
# 配置区
# -----------------------
DATA_DIR = "data"               # 放置 .md/.txt 知识文件的目录
PERSIST_DIR = "chroma_db"       # chroma 持久化目录
CHUNK_SIZE = 500                # 切片大小（字符）
CHUNK_OVERLAP = 100             # 切片重叠（字符）
TOP_K = 3                       # 检索 top-k 值
QWEN_MODEL = "qwen3-max"        # qwen 模型名，按实际替换

# -----------------------
# 1) 读取所有文档
# -----------------------
try:
    file_paths = glob(os.path.join(DATA_DIR, "*.md")) + glob(os.path.join(DATA_DIR, "*.txt"))
    if not file_paths:
        raise FileNotFoundError(f"请在 {DATA_DIR}/ 目录放入示例 .md 或 .txt 文件（UTF-8 编码）")
except Exception as e:
    print(f"❌ 错误：无法读取文档目录 - {e}")
    sys.exit(1)

raw_docs = []
for p in file_paths:
    try:
        # 使用 UTF-8 编码读取文件，若文件不是 UTF-8 请先转码
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            print(f"⚠️  警告：文件 {p} 为空，已跳过")
            continue
        raw_docs.append({"text": text, "source": os.path.basename(p)})
    except UnicodeDecodeError:
        print(f"❌ 错误：文件 {p} 编码不是 UTF-8，请先转换编码")
        sys.exit(1)
    except Exception as e:
        print(f"⚠️  警告：读取文件 {p} 时出错 - {e}，已跳过")
        continue

if not raw_docs:
    print("❌ 错误：没有成功读取任何文档")
    sys.exit(1)

# -----------------------
# 2) 简单切片
# -----------------------
chunks = []      # 存储所有文本片段（字符串）
metadatas = []   # 与 chunks 对应的元数据（例如来源文件名、片段序号）

for doc in raw_docs:
    txt = doc["text"]
    start = 0
    idx = 0
    while start < len(txt):
        end = min(start + CHUNK_SIZE, len(txt))  # 确保不越界
        chunk_text = txt[start:end].strip()  # 去除首尾空白
        
        # 跳过空片段
        if not chunk_text:
            break
            
        # 记录来源和片段索引，便于追溯
        meta = {"source": doc["source"], "chunk_index": idx}
        chunks.append(chunk_text)
        metadatas.append(meta)
        idx += 1
        
        # 下一个片段起始位置（包含重叠，避免重复）
        start = end - CHUNK_OVERLAP if end - CHUNK_OVERLAP > start else end
        if start >= len(txt):  # 防止无限循环
            break

print(f"✅ 已读取 {len(raw_docs)} 个文档，切分为 {len(chunks)} 个片段。")

# -----------------------
# 3) Embedding -> 写入 Chroma（向量化并存储）
# -----------------------
def build_or_load_vectorstore(chunks, metadatas, persist_dir, embedding_model):
    """
    构建或加载向量数据库
    
    如果数据库不存在或数据不完整，则重新构建
    如果数据库存在且完整，则直接加载
    """
    import shutil
    
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        raise ValueError("请设置环境变量 QWEN_API_KEY")
    
    # 检查数据库是否存在
    db_exists = os.path.exists(persist_dir) and os.path.isdir(persist_dir)
    
    if db_exists:
        try:
            # 尝试加载已有数据库
            print(f"📂 检测到已有向量数据库，正在加载...")
            vect = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model
            )
            # 检查数据库是否有数据
            if vect._collection.count() > 0:
                print(f"✅ 成功加载已有向量数据库（包含 {vect._collection.count()} 条记录）")
                return vect
            else:
                print("⚠️  数据库为空，将重新构建...")
                # 删除空数据库
                shutil.rmtree(persist_dir)
        except Exception as e:
            print(f"⚠️  加载数据库失败: {e}，将重新构建...")
            # 删除损坏的数据库
            if os.path.exists(persist_dir):
                shutil.rmtree(persist_dir)
    
    # 构建新数据库
    print(f"🔨 正在构建向量数据库...")
    print(f"   - 文档数量: {len(chunks)}")
    print(f"   - 切片大小: {CHUNK_SIZE} 字符")
    print(f"   - 重叠大小: {CHUNK_OVERLAP} 字符")
    
    vect = Chroma.from_texts(
        texts=chunks, 
        embedding=embedding_model, 
        metadatas=metadatas, 
        persist_directory=persist_dir
    )
    
    # 持久化向量数据库
    try:
        vect.persist()
        print(f"✅ 向量数据库构建完成并已保存到 {persist_dir}")
    except Exception as e:
        print(f"⚠️  警告：向量数据库持久化失败 - {e}")
    
    return vect

try:
    # 初始化 Embedding 模型
    emb = DashScopeEmbeddings(model="text-embedding-v4")
    print("✅ Embedding 模型初始化成功")
    
    # 构建或加载向量数据库
    # 注意：如果数据库已存在且完整，会直接加载；否则会重新构建
    vect = build_or_load_vectorstore(chunks, metadatas, PERSIST_DIR, emb)

except ValueError as e:
    print(f"❌ 配置错误：{e}")
    print("💡 提示：请设置环境变量 QWEN_API_KEY")
    print("   Windows: set QWEN_API_KEY=your_key")
    print("   Linux/Mac: export QWEN_API_KEY=your_key")
    sys.exit(1)
except Exception as e:
    print(f"❌ 错误：向量化失败 - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -----------------------
# 4) 简单检索：使用 vect.as_retriever 并检索 top-k 文档
# -----------------------
try:
    retriever = vect.as_retriever(search_kwargs={"k": TOP_K})
    print(f"✅ 检索器初始化成功（top_k={TOP_K}）")
except Exception as e:
    print(f"❌ 错误：检索器初始化失败 - {e}")
    sys.exit(1)

# -----------------------
# 5) LLM调用
# -----------------------
# 此处复用 only_llm.py 的实现
try:
    from only_llm import chat_qwen
except ImportError:
    print("❌ 错误：无法导入 only_llm 模块，请确保 only_llm.py 文件存在")
    sys.exit(1)

# -----------------------
# 6) 把检索到的片段拼成 prompt
# -----------------------
def build_prompt(query, docs):
    """
    构建 RAG 提示词
    
    Args:
        query: 用户问题
        docs: 检索到的文档列表
    
    Returns:
        构建好的提示词字符串
    """
    if not docs:
        return f"用户问题：{query}\n\n注意：未检索到相关文档，请回答\"我不知道\"。"
    
    parts = []
    for i, d in enumerate(docs, start=1):
        # 兼容不同返回结构（Document对象或字典）
        text = getattr(d, "page_content", str(d))
        src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
        parts.append(f"[片段 {i} | 来源: {src}]\n{text}\n")
    
    context = "\n".join(parts)
    prompt = (
        "下面是检索到的知识片段（可能有冗余），请 **只基于这些片段** 回答问题。\n"
        "如果片段中没有相关信息，请回答\"我不知道\"。\n\n"
        f"已检索到的片段：\n{context}\n\n"
        f"用户问题：{query}\n\n"
        "请给出简洁准确的回答，并在最后列出引用来源（文件名和片段编号）。"
    )
    return prompt

# -----------------------
# 7) RAG查询函数（供外部调用）
# -----------------------
def rag_query(query, retriever_instance=None, model=QWEN_MODEL):
    """
    RAG查询函数
    
    Args:
        query: 用户问题
        retriever_instance: 检索器实例（如果为None，使用全局retriever）
        model: LLM模型名称
    
    Returns:
        dict: 包含answer和sources的字典
    """
    if retriever_instance is None:
        retriever_instance = retriever
    
    try:
        # 检索相关文档
        docs = retriever_instance.invoke(query)
        
        if not docs:
            return {
                "answer": "未检索到相关文档，无法回答。",
                "sources": []
            }
        
        # 构建提示词
        prompt = build_prompt(query, docs)
        
        # 调用 LLM 生成回答
        answer = chat_qwen(prompt, model=model, stream=False)
        
        # 提取来源
        sources = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
            sources.append(f"{src} 片段 {i}")
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"查询失败: {e}",
            "sources": []
        }

# -----------------------
# 8) 运行示例查询
# -----------------------
if __name__ == "__main__":
    query = "公司报销流程是怎样的？"
    print(f"\n{'='*60}")
    print(f"❓ 问题: {query}")
    print(f"{'='*60}\n")
    
    try:
        result = rag_query(query)
        
        # 输出结果
        print(f"\n{'='*60}")
        print("➡️  RAG回答:")
        print(f"{'='*60}")
        print(result["answer"])
        
        # 输出引用来源
        if result["sources"]:
            print(f"\n{'='*60}")
            print("📚 引用来源:")
            print(f"{'='*60}")
            for src in result["sources"]:
                print(f"- {src}")
        
    except Exception as e:
        print(f"❌ 错误：查询失败 - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)