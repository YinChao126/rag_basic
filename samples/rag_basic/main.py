#!/usr/bin/env python3
"""
RAG vs LLM 对比演示

本脚本对比纯LLM和RAG系统回答同一问题的效果差异。
用于快速展示RAG系统的优势。
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 导入模块
try:
    from llm_with_rag import qwen_chat, make_prompt
    from store_manager import QwenEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("💡 提示：请确保已安装所有依赖")
    print("   运行: uv sync")
    sys.exit(1)

# 配置
QWEN_MODEL = "qwen3-max"
TEST_QUESTION = "如何更换电池?"
PERSIST_DIR = "chroma_db"
TOP_K = 3

def print_section(title, char="=", width=70):
    """打印分隔线"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")

def test_llm_only(question):
    """测试纯LLM回答"""
    print("🤖 [方式1] 纯LLM回答（无知识库）")
    print("-" * 70)
    print(f"问题: {question}\n")
    
    try:
        start_time = datetime.now()
        answer = qwen_chat(question, model=QWEN_MODEL, temperature=0.1)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  耗时: {duration:.2f} 秒\n")
        print("回答:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        return {
            "answer": answer,
            "duration": duration,
            "sources": []
        }
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rag(question):
    """测试RAG回答"""
    print("\n📚 [方式2] RAG回答（基于知识库）")
    print("-" * 70)
    print(f"问题: {question}\n")
    
    try:
        # 加载向量数据库
        if not os.path.exists(PERSIST_DIR):
            print(f"❌ 错误：向量数据库不存在: {PERSIST_DIR}")
            print("💡 提示：请先运行 store_manager.py 构建向量数据库")
            return None
        
        emb = QwenEmbeddings()
        vect = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
        retriever = vect.as_retriever(search_kwargs={"k": TOP_K})
        
        # 检索
        print(f"🔍 正在检索（top_k={TOP_K}）...")
        start_time = datetime.now()
        docs = retriever.invoke(question)
        retrieval_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ 检索完成（耗时: {retrieval_time:.3f}秒，检索到{len(docs)}个片段）")
        
        if not docs:
            print("⚠️  警告：未检索到相关文档")
            return None
        
        # 构建Prompt并生成
        prompt = make_prompt(question, docs)
        print(f"🤖 正在生成回答...")
        
        start_time = datetime.now()
        answer = qwen_chat(prompt, model=QWEN_MODEL, temperature=0.1)
        generation_time = (datetime.now() - start_time).total_seconds()
        
        total_time = retrieval_time + generation_time
        
        print(f"⏱️  总耗时: {total_time:.2f} 秒（检索: {retrieval_time:.3f}秒 + 生成: {generation_time:.2f}秒）\n")
        print("回答:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        # 提取来源
        sources = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "unknown") if hasattr(d, "metadata") else "unknown"
            sources.append(f"{src} 片段 {i}")
        
        if sources:
            print("\n📚 引用来源:")
            for src in sources:
                print(f"  - {src}")
        
        return {
            "answer": answer,
            "duration": total_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "sources": sources
        }
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(llm_result, rag_result):
    """对比两种方式的结果"""
    print_section("📊 对比总结", "=")
    
    if not llm_result or not rag_result:
        print("⚠️  无法完成对比，因为某个测试失败")
        return
    
    print(f"{'对比项':<20} {'纯LLM':<25} {'RAG系统':<25}")
    print("-" * 70)
    
    # 回答长度
    llm_len = len(llm_result["answer"])
    rag_len = len(rag_result["answer"])
    print(f"{'回答长度':<20} {llm_len:<25} {rag_len:<25}")
    
    # 响应时间
    llm_time = llm_result["duration"]
    rag_time = rag_result["duration"]
    print(f"{'响应时间':<20} {llm_time:.2f}秒{'':<20} {rag_time:.2f}秒{'':<20}")
    
    # 是否有来源
    llm_sources = "无" if not llm_result["sources"] else f"{len(llm_result['sources'])}个"
    rag_sources = "无" if not rag_result["sources"] else f"{len(rag_result['sources'])}个"
    print(f"{'引用来源':<20} {llm_sources:<25} {rag_sources:<25}")
    
    print("\n" + "=" * 70)
    print("💡 关键差异:")
    print("=" * 70)
    print("✅ RAG系统:")
    print("   - 回答基于知识库内容，更准确可靠")
    print("   - 可以追溯信息来源，便于验证")
    print("   - 能够回答知识库中的特定问题")
    print("   - 通过更新知识库即可更新答案")
    print()
    print("❌ 纯LLM:")
    print("   - 回答基于训练数据，可能不准确")
    print("   - 无法追溯信息来源")
    print("   - 无法回答知识库中的特定问题")
    print("   - 需要重新训练才能更新知识")
    print("=" * 70)
    
    # 保存对比结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "comparison_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RAG vs LLM 对比结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"测试问题: {TEST_QUESTION}\n\n")
        f.write("=" * 70 + "\n")
        f.write("【1】纯LLM回答\n")
        f.write("=" * 70 + "\n")
        f.write(llm_result["answer"])
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("【2】RAG回答\n")
        f.write("=" * 70 + "\n")
        f.write(rag_result["answer"])
        f.write("\n\n")
        if rag_result["sources"]:
            f.write("引用来源:\n")
            for src in rag_result["sources"]:
                f.write(f"  - {src}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("对比总结\n")
        f.write("=" * 70 + "\n")
        f.write(f"回答长度: LLM={llm_len}字符, RAG={rag_len}字符\n")
        f.write(f"响应时间: LLM={llm_time:.2f}秒, RAG={rag_time:.2f}秒\n")
        f.write(f"引用来源: LLM={llm_sources}, RAG={rag_sources}\n")
    
    print(f"\n✅ 对比结果已保存到: {output_file}")

def main():
    """主函数"""
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        print("❌ 错误: 请设置环境变量 QWEN_API_KEY")
        print("   例如: export QWEN_API_KEY='your_api_key'")
        sys.exit(1)
    
    # 打印欢迎信息
    print_section("🚀 RAG vs LLM 对比演示", "=")
    print("本演示将对比纯LLM和RAG系统回答同一问题的效果差异")
    print(f"测试问题: {TEST_QUESTION}")
    print(f"向量数据库: {PERSIST_DIR}")
    
    # 测试纯LLM
    print_section("第一部分：纯LLM回答", "-")
    llm_result = test_llm_only(TEST_QUESTION)
    
    # 等待用户查看
    input("\n按 Enter 键继续查看RAG回答...")
    
    # 测试RAG
    print_section("第二部分：RAG回答", "-")
    rag_result = test_rag(TEST_QUESTION)
    
    # 对比结果
    compare_results(llm_result, rag_result)
    
    print("\n✅ 演示完成！")
    print("\n💡 提示:")
    print("   - 可以修改 TEST_QUESTION 测试其他问题")
    print("   - 可以修改 knowledge/ 目录下的文档添加更多知识")
    print("   - 运行 python llm_with_rag.py 单独测试RAG系统")
    print("   - 运行 python llm_with_rag.py --query '你的问题' 测试自定义问题")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

