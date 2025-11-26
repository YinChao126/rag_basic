"""
把 chunks.json 写入 Chroma 向量数据库。
用法：
  python store_manager.py --chunks chunks.json --persist_dir chroma_db
说明：
  - 本示例用 Chroma.from_texts (自动调用 embeddings) 来构建向量库，简单易懂。
  - 如果 chroma_db 目录存在，会覆盖/更新（取决于 chroma 版本）。
"""

import os
import json
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from pathlib import Path

# 常量定义
EMBED_MODEL = "text-embedding-v4"
DEFAULT_TOP_RERANK = 10

class QwenEmbeddings:
    """阿里云千问嵌入模型包装类"""
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def embed_documents(self, texts):
        """对文档列表进行嵌入"""
        # 检查批量大小，如果超过10个则分批处理
        if len(texts) > DEFAULT_TOP_RERANK:
            embeddings = []
            for i in range(0, len(texts), DEFAULT_TOP_RERANK):
                batch = texts[i:i+DEFAULT_TOP_RERANK]
                res = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
                embeddings.extend([item.embedding for item in res.data])
        else:
            # 调用 embeddings API
            res = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
            embeddings = [item.embedding for item in res.data]
        
        return embeddings

    def embed_query(self, text):
        """对单个查询进行嵌入"""
        res = self.client.embeddings.create(model=EMBED_MODEL, input=[text])
        return res.data[0].embedding

def main():
    import os
    import shutil
    from pathlib import Path
    from datetime import datetime
    
    # 配置参数
    chunks_file = Path("output/chunks.json")
    if not chunks_file.exists():
        chunks_file = Path("chunks.json")
    
    persist_dir = "chroma_db"
    
    print("=" * 70)
    print("💾 向量数据库构建")
    print("=" * 70)
    
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        print("❌ 错误：请设置环境变量 QWEN_API_KEY")
        exit(1)
    
    # 检查切片文件
    if not chunks_file.exists():
        print(f"❌ 错误：找不到切片文件 {chunks_file}")
        print("💡 提示：请先运行 chunker.py 生成切片文件")
        exit(1)
    
    # 检查数据库是否已存在
    if os.path.exists(persist_dir):
        print(f"\n⚠️  检测到已有向量数据库: {persist_dir}")
        response = input("   是否删除并重建？(y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree(persist_dir)
            print(f"   ✅ 已删除旧数据库")
        else:
            print(f"   ⚠️  保留旧数据库，将更新数据")
    
    # 加载切片
    print(f"\n📖 正在加载切片文件...")
    print(f"   文件路径: {chunks_file}")
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [c["text"] for c in chunks]
    metadatas = [{"id": c["id"], "source": c["source"]} for c in chunks]
    
    print(f"✅ 加载成功")
    print(f"   - 切片数量: {len(texts)} 个")
    
    # 构建向量数据库
    print(f"\n🔨 正在构建向量数据库...")
    print(f"   - Embedding模型: text-embedding-v4")
    print(f"   - 数据库路径: {persist_dir}")
    print(f"   - 处理中，请稍候...")
    
    start_time = datetime.now()
    emb = QwenEmbeddings()
    vect = Chroma.from_texts(
        texts=texts, 
        embedding=emb, 
        metadatas=metadatas, 
        persist_directory=persist_dir
    )
    
    # 持久化
    try:
        vect.persist()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 向量数据库构建完成！")
        print(f"   - 耗时: {duration:.2f} 秒")
        print(f"   - 存储路径: {persist_dir}")
        print(f"   - 向量数量: {vect._collection.count() if hasattr(vect, '_collection') else len(texts)}")
        
    except Exception as e:
        print(f"⚠️  持久化警告: {e}")
        print(f"   数据库已构建，但持久化可能失败")
    
    print(f"\n💡 提示：")
    print(f"   - 向量数据库已保存，可以用于检索")
    print(f"   - 运行 llm_with_rag.py 进行RAG问答测试")

if __name__ == "__main__":
    main()



