# gunicorn.conf.py
# Gunicorn 配置文件 - 用于 Kubernetes 生产环境
# 文档: https://docs.gunicorn.org/en/stable/settings.html

import os

# ============================================================
# 基础配置
# ============================================================

# 绑定地址和端口
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8080")

# Worker 类型 - 使用 Uvicorn 的 ASGI worker
worker_class = "uvicorn.workers.UvicornWorker"

# ============================================================
# Worker 配置
# ============================================================

# Worker 数量
# K8s 环境建议: 每个 Pod 2-4 个 worker
# 默认值: 2 (适合 K8s 中 1 核 Pod)
# 可通过环境变量 GUNICORN_WORKERS 覆盖
workers = int(os.getenv("GUNICORN_WORKERS", "2"))

# Worker 最大请求数后重启 (防止内存泄漏)
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "10000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "1000"))

# ============================================================
# 超时配置
# ============================================================

# Worker 超时时间 (秒)
# 如果 worker 在此时间内没有响应，master 会杀死并重启它
# 对于 LLM 工作流，可能需要较长超时
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# 优雅关闭超时 (秒)
# 收到 SIGTERM 后，worker 有这么长时间完成当前请求
# 应小于 K8s terminationGracePeriodSeconds
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "120"))

# Keep-alive 连接超时 (秒)
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# ============================================================
# 预加载配置
# ============================================================

# 预加载应用
# 设为 False: 每个 worker 独立加载应用
# 原因: Apollo 客户端在 Settings 初始化时启动长轮询线程，不能在 fork 前初始化
preload_app = False

# ============================================================
# 日志配置
# ============================================================

# 访问日志 - 输出到 stdout (K8s 友好)
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")

# 错误日志 - 输出到 stderr (K8s 友好)
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")

# 日志级别
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

# 访问日志格式 (JSON 格式便于日志收集)
access_log_format = (
    '{"time": "%(t)s", "remote_addr": "%(h)s", "method": "%(m)s", '
    '"path": "%(U)s", "query": "%(q)s", "status": "%(s)s", '
    '"response_length": "%(B)s", "request_time_us": "%(D)s"}'
)

# 捕获 stdout/stderr 到日志
capture_output = True

# ============================================================
# 进程管理
# ============================================================

# 进程名称前缀
proc_name = "zxy-workflow"

# 工作目录 (仅在 Docker 环境中设置)
# 本地开发时不设置 chdir，使用当前目录
if os.path.exists("/app"):
    chdir = "/app"

# 守护进程模式 (K8s 中必须为 False)
daemon = False
