# FastAPI OpenAI 代理服务器

这是一个 FastAPI 服务器，它代理 OpenAI API 并使用 hexin_engine 后端处理请求。

## 🔑 API Key 认证

服务器使用固定的 API key 进行认证：

```
sk-deeplin-fastapi-proxy-key-12345
```

## 🚀 启动服务器

```bash
# 设置环境变量
export HITHINK_APP_URL="your_app_url"
export HITHINK_APP_ID="your_app_id"
export HITHINK_APP_SECRET="your_app_secret"

# 启动服务器
python deeplin/inference_engine/hexin_server.py
```

服务器将在 `http://localhost:8777` 启动。

## 📖 API 文档

启动服务器后，访问以下链接查看 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 💬 使用 OpenAI 客户端

### 基本用法

```python
import openai

# 创建客户端
client = openai.AsyncOpenAI(
    api_key="sk-deeplin-fastapi-proxy-key-12345",
    base_url="http://localhost:8000/v1"
)

# 列出模型
models = await client.models.list()

# 聊天完成
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

# 流式聊天完成
stream = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
)

async for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
```

### 工具调用

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "北京天气怎么样？"}
    ],
    tools=tools
)
```

## 🧪 测试

### 运行示例客户端

```bash
python tests/test_client.py
```

### 运行集成测试

```bash
# OpenAI 客户端集成测试
python -m pytest tests/test_openai_client_integration.py -v
```

## 🎯 支持的模型

- `gpt-3.5-turbo`
- `gpt-4o`
- `gpt-4o-mini`
- `o3`
- `o4-mini`
- `gpt4`
- `claude`
- `gemini`
- `doubao-deepseek-r1`
- `ep-20250204210426-gclbn`
- `deepseek-reasoner`
- `doubao-deepseek-v3`
- `ep-20250410145517-rpbrz`
- `deepseek-chat`
- `r1-qianfan`

## 📝 注意事项

1. **API Key**: 目前使用固定的 API key，生产环境中应该使用更安全的认证机制
2. **认证**: 服务器启动时会自动获取后端认证 token
3. **模型路由**: 不同模型会被路由到不同的后端服务
4. **错误处理**: 包含完整的错误处理和 HTTP 状态码
5. **流式支持**: 完整支持 OpenAI 流式 API

## 🚨 故障排除

### 认证失败
确保设置了正确的环境变量：
```bash
export HITHINK_APP_URL="..."
export HITHINK_APP_ID="..."
export HITHINK_APP_SECRET="..."
```

### API Key 错误
确保使用正确的 API key：
```
sk-deeplin-fastapi-proxy-key-12345
```

### 连接问题
检查服务器是否在正确的端口运行：
```bash
curl http://localhost:8000/health
```
