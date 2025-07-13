# DeepLin Hexin Server 使用说明

DeepLin 提供了一个 FastAPI 服务器，可以代理 OpenAI API 端点，使用 hexin_engine 作为后端。

## 安装

```bash
pip install deeplin
```

## 使用方法

### 方法 1: 使用模块命令

```bash
# 使用默认端口 (8777)
python -m deeplin.inference_engine.hexin_server

# 使用自定义端口
python -m deeplin.inference_engine.hexin_server --port 9999

# 查看所有可用选项
python -m deeplin.inference_engine.hexin_server --help
```

### 方法 2: 使用脚本命令 (推荐)

```bash
# 使用默认端口 (8777)
deeplin-hexin-server

# 使用自定义端口
deeplin-hexin-server --port 9999

# 查看所有可用选项
deeplin-hexin-server --help
```

## 命令行选项

- `--host HOST`: 绑定的主机地址 (默认: 0.0.0.0)
- `--port PORT`: 绑定的端口 (默认: 8777)
- `--reload`: 启用自动重载 (开发模式)
- `--log-level {debug,info,warning,error}`: 日志级别 (默认: info)

## 环境变量

在启动服务器之前，需要设置以下环境变量：

```bash
export HITHINK_APP_URL="your_app_url"
export HITHINK_APP_ID="your_app_id"
export HITHINK_APP_SECRET="your_app_secret"
```

或者创建一个 `.env` 文件：

```
HITHINK_APP_URL=your_app_url
HITHINK_APP_ID=your_app_id
HITHINK_APP_SECRET=your_app_secret
```

## 示例

启动服务器后，可以通过 OpenAI 兼容的 API 进行调用：

```python
import openai

client = openai.OpenAI(
    api_key="sk-deeplin-fastapi-proxy-key-12345",
    base_url="http://localhost:9999/v1"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

## API 端点

- `GET /v1/models`: 列出可用模型
- `POST /v1/chat/completions`: 创建聊天完成
- `GET /health`: 健康检查端点

## 故障排除

1. **认证失败**: 确保环境变量 `HITHINK_APP_ID` 和 `HITHINK_APP_SECRET` 设置正确
2. **端口被占用**: 使用 `--port` 参数指定不同的端口
3. **模块未找到**: 确保 deeplin 包已正确安装

更多信息请参考项目文档。
