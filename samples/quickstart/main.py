#!/usr/bin/env python3
"""
RAG vs LLM 对比演示

本脚本对比纯LLM和RAG系统回答同一问题的效果差异。
用于快速展示RAG系统的优势。
"""

import os
import sys
from datetime import datetime

# 导入模块
try:
    from only_llm import chat_qwen
    import rag_basic
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 only_llm.py 和 rag_basic.py 在同一目录下")
    sys.exit(1)

# 配置
QWEN_MODEL = "qwen3-max"  # 可根据实际情况修改
TEST_QUESTION = "公司报销流程是怎样的？"

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
        answer = chat_qwen(question, model=QWEN_MODEL, stream=False)
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
        start_time = datetime.now()
        result = rag_basic.rag_query(question, model=QWEN_MODEL)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  耗时: {duration:.2f} 秒\n")
        print("回答:")
        print("-" * 70)
        print(result["answer"])
        print("-" * 70)
        
        if result["sources"]:
            print("\n📚 引用来源:")
            for src in result["sources"]:
                print(f"  - {src}")
        
        return {
            "answer": result["answer"],
            "duration": duration,
            "sources": result["sources"]
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
    print("   - 可以追溯信息来源")
    print("   - 能够回答知识库中的特定问题")
    print()
    print("❌ 纯LLM:")
    print("   - 回答基于训练数据，可能不准确")
    print("   - 无法追溯信息来源")
    print("   - 无法回答知识库中的特定问题")
    print("=" * 70)

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
    print("   - 可以修改 data/faq.md 添加更多知识")
    print("   - 运行 python rag_basic.py 单独测试RAG系统")
    print("   - 运行 python only_llm.py 单独测试纯LLM")

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

