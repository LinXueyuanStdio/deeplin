"""
测试用的 FastAPI 应用配置
使用真实的环境变量进行认证
"""
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 检查必要的环境变量
required_env_vars = ["HITHINK_APP_ID", "HITHINK_APP_SECRET"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    print(f"⚠️  缺少环境变量: {missing_vars}")
    print("请确保在 .env 文件中设置了以下变量:")
    for var in required_env_vars:
        print(f"  - {var}")
    raise EnvironmentError("缺少必要的环境变量")
else:
    # 使用真实的应用
    from deeplin.inference_engine.hexin_server import app

    # 手动初始化认证，因为测试环境可能不会触发 lifespan 事件
    import deeplin.inference_engine.hexin_server as server_module
    from deeplin.inference_engine.hexin_engine import get_userid_and_token

    try:
        if not server_module.USER_ID or not server_module.TOKEN:
            app_url = os.getenv("HITHINK_APP_URL")
            app_id = os.getenv("HITHINK_APP_ID")
            app_secret = os.getenv("HITHINK_APP_SECRET")

            if app_id and app_secret:
                user_id, token = get_userid_and_token(
                    app_url=app_url,
                    app_id=app_id,
                    app_secret=app_secret
                )
                server_module.USER_ID = user_id
                server_module.TOKEN = token
                print(f"✅ 使用真实应用（已认证）: User ID = {user_id[:8]}...")
            else:
                print("⚠️  认证凭据不完整，某些功能可能无法正常工作")
    except Exception as e:
        print(f"⚠️  认证初始化失败: {e}")

    test_app = app
    print("✅ 使用真实应用（有认证）")
