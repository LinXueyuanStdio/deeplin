"""
示例：使用 OpenAI 客户端调用 FastAPI 代理服务器
"""
import asyncio
import openai

# 固定的 API key（与服务器中定义的一致）
API_KEY = "sk-deeplin-fastapi-proxy-key-12345"

# 服务器地址
BASE_URL = "http://localhost:8000/v1"

async def test_openai_client():
    """测试 OpenAI 客户端调用"""

    # 创建 OpenAI 异步客户端
    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    try:
        # 1. 列出可用模型
        print("🔍 获取可用模型...")
        models = await client.models.list()
        print(f"可用模型数量: {len(models.data)}")
        for model in models.data[:3]:  # 显示前3个模型
            print(f"  - {model.id} ({model.owned_by})")

        # 2. 基本聊天完成
        print("\n💬 基本聊天完成...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "你好！请简单介绍一下你自己。"}
            ],
            max_tokens=100
        )
        print(f"回复: {response.choices[0].message.content}")

        # 3. 流式聊天完成
        print("\n🌊 流式聊天完成...")
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "请用一句话描述人工智能的未来。"}
            ],
            max_tokens=50,
            stream=True
        )

        print("流式回复: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # 换行

        # 4. 工具调用示例
        print("\n🔧 工具调用示例...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"}
            ],
            tools=tools,
            max_tokens=100
        )

        if response.choices[0].message.tool_calls:
            print("检测到工具调用:")
            for tool_call in response.choices[0].message.tool_calls:
                print(f"  - {tool_call.function.name}: {tool_call.function.arguments}")
        else:
            print(f"普通回复: {response.choices[0].message.content}")

    except Exception as e:
        print(f"❌ 错误: {e}")

    finally:
        await client.close()

if __name__ == "__main__":
    print("🚀 开始测试 OpenAI 客户端...")
    asyncio.run(test_openai_client())
    print("✅ 测试完成!")
