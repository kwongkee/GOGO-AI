#!/bin/bash

# Celery Worker启动脚本 - 优化版本
cd /app

# 设置环境变量
export PYTHONPATH=/app:$PYTHONPATH
export C_FORCE_ROOT=1

# 停止现有worker
echo "停止现有Celery Worker..."
pkill -f "celery worker" || true
sleep 3

# 启动文件处理Worker
echo "启动文件处理Celery Worker..."
celery -A vectorize worker --loglevel=info --queues=file_processing --concurrency=2 --hostname=worker1@%h --pidfile=/tmp/celery_worker1.pid &

# 启动结构化数据处理Worker  
echo "启动结构化数据Celery Worker..."
celery -A vectorize worker --loglevel=info --queues=structured_data --concurrency=1 --hostname=worker2@%h --pidfile=/tmp/celery_worker2.pid &

# 启动监控Worker
echo "启动监控Celery Worker..."
celery -A vectorize worker --loglevel=info --queues=monitoring --concurrency=1 --hostname=worker3@%h --pidfile=/tmp/celery_worker3.pid &

sleep 5

echo "所有Celery Worker已启动"
echo "使用以下命令检查:"
echo "  ps aux | grep celery"
echo "  celery -A vectorize inspect active"
echo "  celery -A vectorize status"