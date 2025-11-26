import os
import json
import requests
from openai import OpenAI

def chat_qwen(prompt, model, stream=False):
    client = OpenAI(
        api_key=os.environ.get('QWEN_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    print("----- qwen request start -----")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=stream,
        temperature=0.1,
        # top_p=0.9,
    )

    try:
        if stream:
            def generate_content():
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return generate_content()
        else:
            return completion.choices[0].message.content
    except Exception as e:
        print(e)
        return "fail to response"

if __name__ == '__main__':
    import os
    
    # 检查环境变量
    if not os.environ.get("QWEN_API_KEY"):
        print("❌ 错误: 请设置环境变量 QWEN_API_KEY")
        exit(1)
    
    query = "公司报销流程是怎样的？"
    print(f"\n{'='*60}")
    print(f"❓ 问题: {query}")
    print(f"{'='*60}\n")
    
    print("🤖 正在生成回答（纯LLM，无知识库）...")
    answer = chat_qwen(query, "qwen3-max")
    
    print(f"\n{'='*60}")
    print("➡️  回答:")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}")