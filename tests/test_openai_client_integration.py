"""
使用 OpenAI 客户端的集成测试
"""
import asyncio
import sys
import os
import pytest
import pytest_asyncio
import openai
from httpx import AsyncClient
from fastapi.testclient import TestClient

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 使用智能测试应用配置
from test_app import app

# 固定的 API key
API_KEY = "sk-deeplin-fastapi-proxy-key-12345"

class TestOpenAIClientIntegration:
    """使用真实 OpenAI 客户端的集成测试"""

    @pytest_asyncio.fixture
    async def openai_client(self):
        """创建连接到测试服务器的 OpenAI 客户端"""
        # 不 mock 认证信息，使用真实的环境变量
        # 确保服务器会正常初始化认证

        # 创建一个 httpx transport，指向我们的 FastAPI 应用
        from httpx import ASGITransport

        transport = ASGITransport(app=app)

        # 创建 OpenAI 客户端，使用自定义 transport
        client = openai.AsyncOpenAI(
            api_key=API_KEY,
            base_url="http://test/v1",
            http_client=AsyncClient(transport=transport, base_url="http://test")
        )
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_list_models_with_openai_client(self, openai_client):
        """测试使用 OpenAI 客户端列出模型"""
        models = await openai_client.models.list()
        print("=== List Models ===")
        print(models)

        assert models.object == "list"
        assert len(models.data) > 0

        # 验证模型格式
        for model in models.data:
            assert hasattr(model, 'id')
            assert hasattr(model, 'object')
            assert hasattr(model, 'created')
            assert hasattr(model, 'owned_by')
            assert model.object == "model"

    @pytest.mark.asyncio
    async def test_chat_completion_with_openai_client(self, openai_client):
        """测试使用 OpenAI 客户端进行聊天完成"""
        response = await openai_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=50
        )
        print("=== Chat Completion Response ===")
        print(response)

        # 验证响应格式
        assert hasattr(response, 'id')
        assert hasattr(response, 'object')
        assert hasattr(response, 'created')
        assert hasattr(response, 'model')
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')

        assert response.object == "chat.completion"
        assert response.model == "deepseek-reasoner"
        assert len(response.choices) > 0

        # 验证选择格式
        choice = response.choices[0]
        assert hasattr(choice, 'index')
        assert hasattr(choice, 'message')
        assert hasattr(choice, 'finish_reason')

        # 验证消息格式
        message = choice.message
        assert hasattr(message, 'role')
        assert hasattr(message, 'content')
        assert message.role == "assistant"

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_with_openai_client(self, openai_client):
        """测试使用 OpenAI 客户端进行流式聊天完成"""
        stream = await openai_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": "Count from 1 to 3"}
            ],
            max_tokens=30,
            stream=True
        )

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

            # 验证每个 chunk 的格式
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'object')
            assert hasattr(chunk, 'created')
            assert hasattr(chunk, 'model')
            assert hasattr(chunk, 'choices')

            assert chunk.object == "chat.completion.chunk"
            assert chunk.model == "deepseek-reasoner"
            assert len(chunk.choices) > 0

            # 验证选择格式
            choice = chunk.choices[0]
            assert hasattr(choice, 'index')
            assert hasattr(choice, 'delta')
            assert hasattr(choice, 'finish_reason')

        # 应该收到多个 chunks
        assert len(chunks) > 1

        # 最后一个 chunk 应该有 finish_reason
        assert chunks[-1].choices[0].finish_reason is not None

    @pytest.mark.asyncio
    async def test_function_calling_with_openai_client(self, openai_client):
        """测试使用 OpenAI 客户端进行函数调用"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        response = await openai_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}
            ],
            tools=tools,
            max_tokens=100
        )

        # 验证响应格式
        assert response.object == "chat.completion"
        assert len(response.choices) > 0

        # 验证消息可能包含工具调用或普通内容
        message = response.choices[0].message
        assert hasattr(message, 'role')
        assert message.role == "assistant"

        # 如果有工具调用，验证格式
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                assert hasattr(tool_call, 'id')
                assert hasattr(tool_call, 'type')
                assert hasattr(tool_call, 'function')
                assert tool_call.type == "function"

                function = tool_call.function
                assert hasattr(function, 'name')
                assert hasattr(function, 'arguments')

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """测试无效的 API key"""
        from httpx import ASGITransport

        transport = ASGITransport(app=app)
        client = openai.AsyncOpenAI(
            api_key="invalid-key",
            base_url="http://test/v1",
            http_client=AsyncClient(transport=transport, base_url="http://test")
        )

        with pytest.raises(openai.AuthenticationError):
            await client.models.list()

        await client.close()

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        """测试缺少 API key"""
        from httpx import ASGITransport

        transport = ASGITransport(app=app)
        client = openai.AsyncOpenAI(
            api_key="",  # 空的 API key
            base_url="http://test/v1",
            http_client=AsyncClient(transport=transport, base_url="http://test")
        )

        with pytest.raises(openai.AuthenticationError):
            await client.models.list()

        await client.close()

    @pytest.mark.asyncio
    async def test_multiple_models_support(self, openai_client):
        """测试多个模型的支持"""
        models_to_test = ["deepseek-reasoner", "gpt-4o-mini", "claude", "deepseek-chat"]

        for model in models_to_test:
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Say hello"}
                ],
                max_tokens=20
            )

            assert response.object == "chat.completion"
            assert response.model == model
            assert len(response.choices) > 0

if __name__ == "__main__":
    # 运行一个简单的测试示例
    async def simple_test():
        from httpx import ASGITransport

        # 不 mock 认证信息，使用真实的环境变量

        transport = ASGITransport(app=app)
        client = openai.AsyncOpenAI(
            api_key=API_KEY,
            base_url="http://test/v1",
            http_client=AsyncClient(transport=transport, base_url="http://test")
        )

        print("🧪 测试模型列表...")
        models = await client.models.list()
        print(f"✅ 成功获取 {len(models.data)} 个模型")

        print("🧪 测试聊天完成...")
        try:
            response = await client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=20
            )
            print(f"✅ 成功获取回复: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            if "500" in str(e) or "timeout" in str(e).lower():
                print(f"✅ 正确处理了后端错误: {str(e)[:50]}...")
            else:
                print(f"❌ 意外错误: {e}")

        await client.close()

    print("🚀 运行简单测试...")
    asyncio.run(simple_test())
    print("✅ 简单测试完成!")
