import os
import json
from openai import OpenAI
from langchain_community.embeddings import OllamaEmbeddings
from pathlib import Path

# 常量定义
EMBED_MODEL = "text-embedding-v4"
DEFAULT_TOP_RERANK = 10

def embed_text(texts: any, model: str = EMBED_MODEL):
    """
    将单个字符串或字符串列表转为向量。
    返回：如果输入是字符串，返回 list(float)；如果输入是 list[str]，返回 list[list[float]]
    使用 Aliyun-compatible OpenAI SDK (OpenAI class) to call embeddings.create.
    """
    client = OpenAI(api_key=os.environ.get("QWEN_API_KEY"),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    single_input = False
    if isinstance(texts, str):
        texts = [texts]
        single_input = True

    # 检查批量大小，如果超过10个则分批处理
    if len(texts) > DEFAULT_TOP_RERANK:
        embeddings = []
        for i in range(0, len(texts), DEFAULT_TOP_RERANK):
            batch = texts[i:i+DEFAULT_TOP_RERANK]
            res = client.embeddings.create(model=model, input=batch)
            embeddings.extend([item.embedding for item in res.data])
    else:
        # 调用 embeddings API
        res = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in res.data]

    return embeddings[0] if single_input else embeddings

def main():
    import os
    from pathlib import Path
    from datetime import datetime
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        print("❌ 错误：请设置环境变量 QWEN_API_KEY")
        exit(1)
    
    # 配置参数
    chunks_file = output_dir / "chunks.json"
    if not chunks_file.exists():
        chunks_file = Path("chunks.json")
    
    if not chunks_file.exists():
        print(f"❌ 错误：找不到切片文件 {chunks_file}")
        print("💡 提示：请先运行 chunker.py 生成切片文件")
        exit(1)
    
    print("=" * 70)
    print("🔢 文本向量化测试")
    print("=" * 70)
    
    # 加载切片
    print(f"\n📖 正在加载切片文件...")
    print(f"   文件路径: {chunks_file}")
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    
    print(f"✅ 加载成功")
    print(f"   - 切片数量: {len(texts)} 个")
    print(f"   - 平均长度: {sum(len(t) for t in texts) / len(texts):.0f} 字符")
    
    # ========== 使用阿里云模型 ==========
    print(f"\n{'='*70}")
    print("方法1：使用阿里云 text-embedding-v4 模型")
    print(f"{'='*70}")
    
    batch_size = 10
    print(f"📝 开始向量化...")
    print(f"   - 模型: {EMBED_MODEL}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 预计批次: {(len(texts) + batch_size - 1) // batch_size} 批")
    
    start_time = datetime.now()
    vectors = embed_text(texts, model=EMBED_MODEL)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if len(vectors) != len(texts):
        raise RuntimeError("embeddings 数量与文本数量不一致")
    
    print(f"✅ 向量化完成！")
    print(f"   - 耗时: {duration:.2f} 秒")
    print(f"   - 向量维度: {len(vectors[0])} 维")
    print(f"   - 处理速度: {len(texts)/duration:.1f} 个/秒")
    
    # 保存结果
    output_file = output_dir / "embeddings.jsonl"
    with open(output_file, "w", encoding="utf-8") as fout:
        for i, vec in enumerate(vectors):
            item = {
                "id": ids[i],
                "source": chunks[i]["source"],
                "embedding_len": len(vec),
                "text_preview": texts[i][:120].replace("\n", " "),
                "embedding": vec
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"   - 输出文件: {output_file}")
    
    # ========== 使用Ollama模型（可选）==========
    print(f"\n{'='*70}")
    print("方法2：使用本地 Ollama 模型（可选对比）")
    print(f"{'='*70}")
    
    try:
        print(f"📝 尝试连接本地Ollama服务...")
        emb_ollama = OllamaEmbeddings(model="all-minilm:latest")
        
        print(f"✅ 连接成功")
        print(f"   - 模型: all-minilm:latest")
        
        start_time = datetime.now()
        vectors_ollama = emb_ollama.embed_documents(texts)
        end_time = datetime.now()
        duration_ollama = (end_time - start_time).total_seconds()
        
        if len(vectors_ollama) != len(texts):
            raise RuntimeError("embeddings 数量与文本数量不一致")
        
        print(f"✅ 向量化完成！")
        print(f"   - 耗时: {duration_ollama:.2f} 秒")
        print(f"   - 向量维度: {len(vectors_ollama[0])} 维")
        print(f"   - 处理速度: {len(texts)/duration_ollama:.1f} 个/秒")
        
        # 保存结果
        output_file_ollama = output_dir / "embeddings_ollama.jsonl"
        with open(output_file_ollama, "w", encoding="utf-8") as fout:
            for i, vec in enumerate(vectors_ollama):
                item = {
                    "id": ids[i],
                    "source": chunks[i]["source"],
                    "embedding_len": len(vec),
                    "text_preview": texts[i][:120].replace("\n", " "),
                    "embedding": vec
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"   - 输出文件: {output_file_ollama}")
        
        # 对比总结
        print(f"\n{'='*70}")
        print("📊 向量化模型对比")
        print(f"{'='*70}")
        print(f"{'模型':<30} {'维度':<15} {'耗时':<15} {'速度':<15}")
        print("-" * 70)
        print(f"{'text-embedding-v4 (云端)':<30} {len(vectors[0]):<15} {duration:.2f}秒{'':<10} {len(texts)/duration:.1f}个/秒")
        print(f"{'all-minilm:latest (本地)':<30} {len(vectors_ollama[0]):<15} {duration_ollama:.2f}秒{'':<10} {len(texts)/duration_ollama:.1f}个/秒")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"⚠️  本地Ollama模型不可用: {e}")
        print("💡 提示：可以跳过此步骤，使用云端模型即可")
    
    print(f"\n✅ 向量化流程完成！")
    print(f"💡 提示：向量化后的数据将用于构建向量数据库")

if __name__ == "__main__":
    main()



