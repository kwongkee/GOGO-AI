# GOGO-AI Dockerfile
# Python 3.9 + AI客服服务

FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    curl \
    wget \
    git \
    vim \
    netcat \
    kafka-manager \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY 国内服务器/ /app/
COPY 美国服务器/ /app/

# 创建必要目录
RUN mkdir -p /app/logs /app/sql /app/output

# 设置权限
RUN chmod +x /app/*.sh 2>/dev/null || true

# 暴露端口
EXPOSE 5000 8000 9090

# 启动命令（默认启动Kafka消费者）
CMD ["python", "kafka_manager.py"]
