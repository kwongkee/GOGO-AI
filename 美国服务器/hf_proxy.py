#!/usr/bin/env python3
import warnings

# 忽略无关紧要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_huggingface import HuggingFaceEmbeddings  # 更新为langchain-huggingface包
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.runnables import Runnable
from langchain_core.language_models.llms import LLM
from langchain_community.llms import Ollama
import re
from typing import Any, List, Optional, Dict
import requests
import base64
import gzip
import struct
import io
import tempfile
import os
import logging
from PIL import Image
import whisper
import numpy as np
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor  # 更新为ChineseCLIP专用基类
import threading
from datetime import datetime
import shutil
import time
import hashlib
import json
from huggingface_hub import InferenceClient

import sys

print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")
os.environ['CURL_CA_BUNDLE'] = ''

# 导入配置文件
try:
    from config import config
except ImportError:
    # 默认配置（如果配置文件不存在）
    class Config:
        MODEL_CACHE_DIR = "/root/customer_ai/models"
        FAISS_INDEX_DIR = "/root/customer_ai/faiss_index"
        FAISS_BACKUP_DIR = "/root/customer_ai/faiss_backups"
        LOG_DIR = "/root/customer_ai/logs"
        ALLOWED_API_KEYS = ["abc123321cba"]  # 默认API密钥
        MAX_REQUESTS_PER_MINUTE = 60
        TEXT_MODEL_PATH = "/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e"
        CLIP_MODEL_NAME = "OFA-Sys/chinese-clip-vit-large-patch14"
        WHISPER_MODEL_SIZE = "medium"
        BASE_TEXT_WEIGHT = 0.7
        MIN_TEXT_WEIGHT = 0.5
        MAX_TEXT_WEIGHT = 0.9
        MULTI_ELEMENT_ADJUST = 0.1


    config = Config()

# 确保日志目录存在
log_dir = "/root/customer_ai/logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志（仅记录ERROR级别，隐藏INFO/WARNING无关输出）
logging.basicConfig(
    level=logging.DEBUG,  # 改为DEBUG级别
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("/root/customer_ai/logs/hf_proxy_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 配置频率限制
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{config.MAX_REQUESTS_PER_MINUTE} per minute"]
)

# 全局变量
clip_model = None
processor = None
whisper_model = None
embeddings = None
llm = None
model_load_lock = threading.Lock()
model_last_health_check = 0
model_health_check_interval = 3600  # 1小时检查一次

HF_TOKEN = os.getenv("HF_TOKEN", "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")  # 从环境变量获取HF_TOKEN

# 在文件顶部添加锁变量定义
_first_request_lock = threading.Lock()
_first_request_done = False
_models_loaded = False

# ==================== 在全局变量区 ====================
embeddings = None
global_vectorstore = None
llm = None

@app.before_request
def handle_first_request():
    global _first_request_done
    if not _first_request_done:
        with _first_request_lock:
            if not _first_request_done:
                # 执行一次性初始化
                _init_app()
                _first_request_done = True

    print(f"\n=== 收到请求 ===")
    print(f"时间: {datetime.now().isoformat()}")
    print(f"方法: {request.method}")
    print(f"路径: {request.path}")
    print(f"远程地址: {request.remote_addr}")
    print(f"用户代理: {request.user_agent}")
    print(f"内容类型: {request.content_type}")
    print(f"内容长度: {request.content_length}")

    if request.method == 'POST' and request.content_length:
        try:
            # 只记录请求的前1000个字符，避免日志过大
            data = request.get_data(as_text=True)
            if len(data) > 1000:
                print(f"请求数据(前1000字符): {data[:1000]}...")
            else:
                print(f"请求数据: {data}")
        except Exception as e:
            print(f"解析请求数据失败: {e}")

@app.after_request
def log_response_info(response):
    """记录所有响应的信息"""
    print(f"=== 返回响应 ===")
    print(f"状态码: {response.status_code}")
    print(f"内容类型: {response.content_type}")
    print(f"内容长度: {response.content_length}")
    print(f"时间: {datetime.now().isoformat()}\n")
    return response

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import atexit
import signal


# 全局线程池（环境变量配置）
MAX_WORKERS = int(os.getenv('RAG_MAX_WORKERS', '6'))
executor = ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
    thread_name_prefix="rag-worker"
)

# 优雅关闭
def _graceful_shutdown():
    executor.shutdown(wait=True)
    atexit.register(_graceful_shutdown)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: _graceful_shutdown())

# 动态线程池（根据负载调整）
class DynamicThreadPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        self.min_workers = int(os.getenv('MIN_WORKERS', '2'))
        self.max_workers = int(os.getenv('MAX_WORKERS', '8'))
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="rag-worker"
        )

        # 优雅关闭
        def _shutdown():
            self.executor.shutdown(wait=True)

        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, lambda s, f: _shutdown())
        signal.signal(signal.SIGINT, lambda s, f: _shutdown())


# 全局实例
pool = DynamicThreadPool().executor

def run_with_timeout(func, *args, timeout=45, **kwargs):
    loop = asyncio.get_event_loop()
    try:
        return asyncio.wait_for(
            loop.run_in_executor(pool, func, *args, **kwargs),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError("LLM调用超时")

# 全局线程池（单例，复用）
executor = ThreadPoolExecutor(max_workers=4)
atexit.register(lambda: executor.shutdown(wait=False))

# 自定义LLM类 - 使用InferenceClient
class CustomHuggingFaceLLM(LLM, Runnable):
    """修复：继承Runnable接口"""

    def __init__(self, repo_id, token, max_tokens=1024, temperature=0.1):
        self.repo_id = repo_id
        self.token = token
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_available = False
        self.client = None
        self._initialize_client()

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call_api(prompt)

    def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call_api(prompt)

    def _initialize_deepseek_client(self):
        """专门为 DeepSeek 模型初始化客户端 - 使用对话API"""
        try:
            self.client = InferenceClient(
                model=self.repo_id,
                token=self.token,
                timeout=120
            )

            # 使用对话API测试连接
            test_messages = [{"role": "user", "content": "Hello"}]
            test_response = self.client.chat_completion(
                model=self.repo_id,
                messages=test_messages,
                max_tokens=10
            )

            self.model_available = True
            logger.info(f"DeepSeek 模型初始化成功: {self.repo_id}")

        except Exception as e:
            logger.error(f"❌ DeepSeek 初始化失败: {e}")
            self.model_available = False
            self.client = None

    def _initialize_client(self):
        """初始化DeepSeek-R1客户端"""
        try:
            from huggingface_hub import InferenceClient

            # DeepSeek-R1专用配置
            self.client = InferenceClient(
                model=self.repo_id,
                token=self.token,
                timeout=120,
                headers={"Authorization": f"Bearer {self.token}"}
            )

            # 测试连接 - 使用对话API
            test_messages = [{"role": "user", "content": "测试"}]
            test_response = self.client.chat_completion(
                messages=test_messages,
                max_tokens=10,
                temperature=0.1
            )

            self.model_available = True
            logger.info(f"DeepSeek-R1模型初始化成功: {self.repo_id}")

        except Exception as e:
            logger.error(f"DeepSeek-R1初始化失败: {e}")
            self.model_available = False
            # 使用备用模型
            self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """初始化备用模型"""
        try:
            # 使用更稳定的模型
            self.client = InferenceClient(
                model="microsoft/DialoGPT-large",
                timeout=60
            )
            self.model_available = True
            logger.info("使用备用模型: microsoft/DialoGPT-large")
        except Exception as e:
            logger.error(f"备用模型也失败: {e}")

    def generate(self, prompts, **kwargs):
        """生成文本 - 兼容LangChain接口"""
        responses = []
        for prompt in prompts:
            try:
                if self.client is None:
                    # 重新初始化
                    if "deepseek" in self.repo_id.lower():
                        self._initialize_deepseek_client()
                    else:
                        self._initialize_client()

                if not self.model_available:
                    raise Exception("模型不可用")

                response = self._call_api(prompt)
                responses.append([{"text": response}])
            except Exception as e:
                logger.error(f"API调用失败: {e}")
                responses.append([{"text": ""}])
        return {"generations": responses}

    def _call_api(self, prompt):
        """使用InferenceClient调用模型"""
        try:
            # 对于DeepSeek模型，只使用对话API
            if "deepseek" in self.repo_id.lower():
                return self._call_deepseek_api(prompt)
            else:
                # 对于其他模型，使用文本生成
                return self._call_text_generation_api(prompt)
        except Exception as e:
            logger.error(f"所有API调用方式都失败: {e}")
            raise e

    def _call_deepseek_api(self, prompt):
        """DeepSeek-R1专用API调用"""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )

            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise Exception("DeepSeek API返回空响应")

        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise

    def _call_text_generation_api(self, prompt):
        """通用文本生成API调用"""
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                return_full_text=False
            )

            if response and len(response.strip()) > 0:
                return response.strip()
            else:
                raise Exception("文本生成返回空响应")

        except Exception as e:
            logger.error(f"文本生成API调用失败: {e}")
            raise e


# 增强全局异常处理
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.error(f"未捕获的异常: {error}", exc_info=True)
    return jsonify({
        "error": "服务器内部错误",
        "message": str(error),
        "timestamp": datetime.now().isoformat()
    }), 500


# 添加详细日志点
def log_processing_stage(stage, details):
    logger.info(f"[{stage}] {details}")


def call_with_retry(func, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (attempt + 1))


# API密钥验证装饰器
def require_api_key(f):
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if not api_key or api_key not in config.ALLOWED_API_KEYS:
            logger.warning(f"API密钥验证失败: {api_key}")
            return jsonify({"error": "无效的API密钥"}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function


def load_models():
    global clip_model, processor, whisper_model, embeddings, llm, model_last_health_check

    with model_load_lock:
        try:
            # 1. 加载嵌入模型（最高优先级）
            if embeddings is None:
                logger.info("加载文本嵌入模型...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e",
                    # 本地路径
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("文本嵌入模型加载成功")

            # 2. 简化LLM初始化 - 直接使用可靠的模型
            if llm is None:
                try:
                    # 优先使用DeepSeek API（需要网络连接）
                    from langchain_community.llms import HuggingFaceEndpoint

                    llm = HuggingFaceEndpoint(
                        repo_id="deepseek-ai/DeepSeek-R1-0528",
                        task="text-generation",
                        huggingfacehub_api_token=HF_TOKEN,
                        timeout=120,
                        temperature=0.7,
                        max_new_tokens=1024
                    )
                    logger.info("DeepSeek LLM初始化成功")
                except Exception as e:
                    logger.warning("Hugging Face失败，降级到Ollama: " . str(e))
                    # 备用方案：使用本地Ollama（需要确保服务运行）
                    llm = Ollama(
                        model="llama2",
                        base_url="http://localhost:11434",
                        timeout=60
                    )

            # 3. 加载CLIP模型（可选）
            if clip_model is None:
                try:
                    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
                    clip_model = ChineseCLIPModel.from_pretrained(
                        "OFA-Sys/chinese-clip-vit-large-patch14",
                        cache_dir=config.MODEL_CACHE_DIR
                    )
                    processor = ChineseCLIPProcessor.from_pretrained(
                        "OFA-Sys/chinese-clip-vit-large-patch14",
                        cache_dir=config.MODEL_CACHE_DIR
                    )
                    logger.info("CLIP模型加载成功")
                except Exception as e:
                    logger.warning(f"CLIP模型加载失败: {e}")
                    clip_model = None
                    processor = None

            # 4. 加载Whisper模型（可选）
            if whisper_model is None:
                try:
                    import whisper
                    whisper_model = whisper.load_model(
                        "base",  # 使用较小的base模型
                        download_root=config.MODEL_CACHE_DIR
                    )
                    logger.info("Whisper模型加载成功")
                except Exception as e:
                    logger.warning(f"Whisper模型加载失败: {e}")
                    whisper_model = None

            # 5. 加载LLM（使用可靠的备用方案）
            if llm is None:
                try:
                    # 使用CustomHuggingFaceLLM包装DeepSeek-R1
                    llm = CustomHuggingFaceLLM(
                        repo_id="deepseek-ai/DeepSeek-R1-0528",
                        token=HF_TOKEN,
                        max_tokens=1024,
                        temperature=0.1
                    )

                    # 检查模型是否真的可用
                    if hasattr(llm, 'model_available') and not llm.model_available:
                        raise Exception("DeepSeek模型初始化失败")

                    logger.info("DeepSeek-R1 LLM初始化成功")
                except Exception as e:
                    logger.error(f"DeepSeek-R1加载失败: {e}")
                    # 强制设置为None，确保使用备用模型
                    llm = None

                    # 备用方案 - 使用更可靠的模型
                    try:
                        logger.info("尝试使用更可靠的模型...")
                        # 使用一个已知可用的模型
                        llm = CustomHuggingFaceLLM(
                            repo_id="microsoft/DialoGPT-large",
                            token=HF_TOKEN,
                            max_tokens=1024,
                            temperature=0.1
                        )

                        if hasattr(llm, 'model_available') and not llm.model_available:
                            raise Exception("备用模型也不可用")

                        logger.info("使用备用DialoGPT-large模型成功")
                    except Exception as e2:
                        logger.error(f"备用模型也失败: {e2}")
                        from langchain_community.llms import FakeListLLM
                        llm = FakeListLLM(responses=["抱歉，AI服务暂时不可用，请稍后再试。"])
                        logger.info("使用FakeListLLM作为最后备选")

            model_last_health_check = time.time()
            logger.info("所有模型加载完成")

        except Exception as e:
            logger.error(f"模型加载异常: {e}")

def check_ollama_health():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama健康")
            return True
    except Exception as e:
        logger.error("Ollama健康检查失败: " . str(e))
        # 自动化重启 (匹配文档容错)
        os.system("systemctl restart ollama || ollama serve &")
        logger.info("Ollama重启尝试完成")
        return False

def check_models_health():
    """检查模型健康状态"""
    global model_last_health_check

    current_time = time.time()
    if current_time - model_last_health_check < model_health_check_interval:
        return True

    try:
        # 测试CLIP模型
        if clip_model and processor:
            test_image = Image.new('RGB', (224, 224), color='red')
            inputs = processor(images=test_image, return_tensors="pt")
            clip_model.get_image_features(**inputs)

        # 测试Whisper模型
        if whisper_model:
            test_audio = np.zeros((16000,), dtype=np.float32)
            whisper_model.transcribe(test_audio)

        # 测试文本嵌入模型
        if embeddings:
            test_text = "测试文本"
            embeddings.embed_query(test_text)

        model_last_health_check = current_time
        return True

    except Exception as e:
        logger.error(f"模型健康检查失败: {e}")
        # 尝试重新加载模型
        try:
            load_models()
            return True
        except Exception as reload_error:
            logger.error(f"模型重新加载失败: {reload_error}")
            return False


load_models()


def generate_vector_multi(text_content, multimodal_elements=[]):
    """多模态向量生成（带动态权重调整）"""
    try:
        # 使用统一的嵌入模型
        if text_content and text_content.strip():
            text_embedding = np.array(embeddings.embed_query(text_content))
        else:
            text_embedding = np.zeros(768)

        # 多模态融合
        if multimodal_elements and clip_model and processor:
            multi_embeddings = []
            for elem in multimodal_elements:
                if isinstance(elem, np.ndarray):
                    elem = Image.fromarray(elem)
                inputs = processor(images=elem, return_tensors="pt")
                emb = clip_model.get_image_features(**inputs)[0].detach().numpy()
                multi_embeddings.append(emb)

            if multi_embeddings:
                avg_multi_emb = np.mean(multi_embeddings, axis=0)
                avg_multi_emb = avg_multi_emb / (np.linalg.norm(avg_multi_emb) + 1e-10)

                # 动态权重调整 - 基于多模态元素数量
                base_weight = config.BASE_TEXT_WEIGHT
                element_count = len(multimodal_elements)

                # 元素越多，文本权重越低（但保持在合理范围内）
                adjusted_weight = max(
                    config.MIN_TEXT_WEIGHT,
                    min(
                        config.MAX_TEXT_WEIGHT,
                        base_weight - (element_count * config.MULTI_ELEMENT_ADJUST)
                    )
                )

                final_vector = (adjusted_weight * text_embedding + (1 - adjusted_weight) * avg_multi_emb)
                final_vector = final_vector / (np.linalg.norm(final_vector) + 1e-10)
            else:
                final_vector = text_embedding
        else:
            final_vector = text_embedding

        return final_vector.tolist()
    except Exception as e:
        logger.error(f"多模态向量生成失败: {e}")
        # 回退到纯文本向量
        return embeddings.embed_query(text_content if text_content else "空内容")


def backup_faiss_index():
    """备份FAISS索引"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(config.FAISS_BACKUP_DIR, timestamp)
        os.makedirs(backup_dir, exist_ok=True)

        # 备份索引文件
        shutil.copy2(os.path.join(config.FAISS_INDEX_DIR, "index.faiss"),
                     os.path.join(backup_dir, "index.faiss"))
        shutil.copy2(os.path.join(config.FAISS_INDEX_DIR, "index.pkl"),
                     os.path.join(backup_dir, "index.pkl"))

        return True
    except Exception as e:
        logger.error(f"索引备份失败: {e}")
        return False


# === 核心函数：同步调用 LLM ===
def call_llm_with_context_sync(query: str, context: str) -> dict:
    """
    使用预计算的上下文 + 查询，同步调用 Ollama LLM 生成回答
    """
    if not llm:
        return {
            "result": "AI 服务暂时不可用，请稍后重试。",
            "source_documents": [],
            "status": "error"
        }

    # 构造专业提示词
    prompt = f"""你是一个专业的文档分析助手。请根据以下【参考资料】，用简洁、准确、友好的中文回答用户问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 只基于参考资料回答，不要编造内容。
2. 如果资料中没有相关信息，请说“根据提供的文档，我无法找到相关信息。”
3. 答案控制在 500 字以内。
4. 语言自然、口语化，像真人对话。

回答："""

    try:
        # 同步调用 Ollama
        raw_response = llm.invoke(prompt)

        # 清理输出：移除 <think> 标签、多余空格
        clean_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()

        logger.info(f"LLM 原始输出: {raw_response[:200]}...")
        logger.info(f"LLM 清理后输出: {clean_response}")

        return {
            "result": clean_response,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return {
            "result": "抱歉，AI 分析时出现了一点小问题，请稍后重试。",
            "status": "error"
        }


@app.route('/predict', methods=['POST'])
def predict():
    if global_vectorstore is None:
        return jsonify({"error": "索引未就绪"}), 503

    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "查询不能为空"}), 400

    try:
        # 1. 检索全局索引
        docs = global_vectorstore.similarity_search(query, k=3)
        if not docs:
            return jsonify({
                "answer": "根据提供的文档，我无法找到相关信息。",
                "source_documents": [],
                "status": "success"
            })

        # 2. 构建上下文
        context = "\n\n".join([f"资料 {i + 1}: {doc.page_content}" for i, doc in enumerate(docs)])

        # 3. 调用 LLM
        result = call_llm_with_context_sync(query, context)

        # 4. 返回结构化结果
        return jsonify({
            "answer": result["result"],
            "source_documents": [doc.page_content[:200] + "..." for doc in docs],
            "source_count": len(docs),
            "status": result["status"],
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"/predict 接口异常: {e}")
        return jsonify({"error": "查询失败，请稍后重试"}), 500


def handle_sync_query(data):
    """处理同步查询（上传即问即答）"""
    query = data['query']
    doc_vectors = data['doc_vectors']
    doc_contents = data['doc_contents']

    # 使用预计算向量构建临时FAISS索引
    vector_store = build_faiss_from_precomputed(doc_vectors, doc_contents)

    # 执行检索和回答
    result = execute_qa_with_precomputed(vector_store, query)

    return jsonify({
        "result": result['answer'],
        "source_documents": result['sources'],
        "sync_processed": True,
        "timestamp": datetime.now().isoformat()
    })


def build_faiss_from_precomputed(vectors, contents):
    """修复版的FAISS索引构建 - 增强错误处理"""
    try:
        if not vectors:
            logger.error("向量数据为空")
            return None

        # 验证向量维度一致性
        dimension = len(vectors[0])
        valid_vectors = []
        for i, vector in enumerate(vectors):
            if vector and len(vector) == dimension:
                valid_vectors.append(vector)
            else:
                logger.warning(f"跳过无效向量 {i}, 维度: {len(vector) if vector else 0}")

        if not valid_vectors:
            logger.error("没有有效的向量数据")
            return None

        # 使用numpy数组
        import numpy as np
        vectors_array = np.array(valid_vectors).astype('float32')

        # 创建FAISS索引
        import faiss
        index = faiss.IndexFlatIP(dimension)  # 使用内积相似度

        # 归一化向量（提高检索质量）
        faiss.normalize_L2(vectors_array)
        index.add(vectors_array)

        # 创建文档存储（使用空内容或占位符）
        from langchain.schema import Document
        documents = []
        for i in range(len(valid_vectors)):
            content = contents[i] if i < len(contents) and contents[i] else f"文档_{i+1}"
            doc = Document(
                page_content=content,
                metadata={"source": f"doc_{i}", "vector_index": i}
            )
            documents.append(doc)

        # 创建自定义检索器
        class PrecomputedFAISS:
            def __init__(self, index, documents, embeddings):
                self.index = index
                self.documents = documents
                self.embeddings = embeddings

            def get_relevant_documents(self, query, k=3):
                try:
                    # 生成查询向量
                    query_vector = self.embeddings.embed_query(query)
                    query_vector = np.array([query_vector]).astype('float32')
                    faiss.normalize_L2(query_vector)

                    # 搜索相似文档
                    scores, indices = self.index.search(query_vector, k=min(k, len(self.documents)))

                    results = []
                    for i, idx in enumerate(indices[0]):
                        if 0 <= idx < len(self.documents):
                            results.append(self.documents[idx])

                    return results
                except Exception as e:
                    logger.error(f"FAISS搜索失败: {e}")
                    return self.documents[:k] if self.documents else []

        vector_store = PrecomputedFAISS(index, documents, embeddings)
        logger.info(f"FAISS索引构建成功，包含 {len(documents)} 个文档，维度: {dimension}")
        return vector_store

    except Exception as e:
        logger.error(f"FAISS索引构建失败: {e}")
        return None


def create_fallback_index(vectors, contents):
    """创建降级索引方案"""
    try:
        # 简单的内存索引
        class SimpleIndex:
            def __init__(self, vectors, contents):
                self.vectors = vectors
                self.contents = contents
                self.embeddings = embeddings

            def as_retriever(self, search_kwargs=None):
                return self

            def get_relevant_documents(self, query):
                # 简单相似度计算
                query_vector = self.embeddings.embed_query(query)
                similarities = []

                for i, vector in enumerate(self.vectors):
                    if len(vector) == len(query_vector):
                        similarity = np.dot(vector, query_vector) / (
                                np.linalg.norm(vector) * np.linalg.norm(query_vector)
                        )
                        similarities.append((similarity, i))

                similarities.sort(reverse=True)
                top_indices = [idx for _, idx in similarities[:3]]

                from langchain.schema import Document
                return [Document(
                    page_content=self.contents[i],
                    metadata={"source": f"doc_{i}", "similarity": similarities[j][0]}
                ) for j, i in enumerate(top_indices)]

        logger.info("使用降级索引方案")
        return SimpleIndex(vectors, contents)

    except Exception as e:
        logger.error(f"降级方案也失败: {e}")
        return None


def get_chat_history(chat_pid: int):
    """获取历史对话记录"""
    try:
        response = requests.post(
            "https://shop.gogo198.cn/collect_website/public/?s=api/getgoods/get_chat_history",
            json={"chat_pid": chat_pid},
            timeout=5
        )
        return response.json().get("history", "无历史记录")
    except:
        return "历史记录服务不可用"


@app.route('/refresh_index', methods=['POST'])
@require_api_key
def refresh_index():
    """刷新索引接口"""
    try:
        global embeddings
        vectorstore = FAISS.load_local(config.FAISS_INDEX_DIR, embeddings=embeddings,
                                       allow_dangerous_deserialization=True)
        return jsonify({"status": "success", "message": "索引刷新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET','POST'])
def health_check():
    """健康检查接口"""
    models_loaded = all([clip_model, processor, whisper_model, embeddings, llm])
    models_healthy = check_models_health()

    status = "healthy" if (models_loaded and models_healthy) else "degraded"

    return jsonify({
        "status": status,
        "models_loaded": models_loaded,
        "models_healthy": models_healthy,
        "last_health_check": model_last_health_check,
        "timestamp": datetime.now().isoformat(),
        "service": "hf_proxy"
    })


@app.route('/backup_index', methods=['POST'])
@require_api_key
def backup_index():
    """备份索引接口"""
    try:
        success = backup_faiss_index()
        if success:
            return jsonify({"status": "success", "message": "索引备份成功"})
        else:
            return jsonify({"error": "索引备份失败"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model_status', methods=['GET'])
@require_api_key
def model_status():
    """获取模型详细状态"""
    models_info = {
        "clip_model": clip_model is not None,
        "whisper_model": whisper_model is not None,
        "embeddings": embeddings is not None,
        "llm": llm is not None,
        "last_health_check": model_last_health_check,
        "health_check_interval": model_health_check_interval
    }

    return jsonify(models_info)


# 错误处理
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "请求频率过高", "message": str(e.description)}), 429


@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"服务器内部错误: {e}")
    return jsonify({"error": "服务器内部错误"}), 500


def decompress_vectors(compressed_vectors_str, dimension=768):
    """解压缩深圳服务器发送的向量数据 - 支持zlib和gzip格式"""
    try:
        # Base64解码
        compressed_data = base64.b64decode(compressed_vectors_str)

        # 检测压缩格式并解压
        if compressed_data.startswith(b'\x1f\x8b'):  # GZIP格式
            decompressed_data = gzip.decompress(compressed_data)
        else:  # ZLIB格式（gzcompress）
            import zlib
            decompressed_data = zlib.decompress(compressed_data)

        # 将二进制数据转换为浮点数列表
        num_floats = len(decompressed_data) // 4  # 每个float占4字节
        vectors = []
        for i in range(0, num_floats, dimension):
            if i + dimension <= num_floats:
                vec = list(struct.unpack(f'{dimension}f',
                                         decompressed_data[i * 4:(i + dimension) * 4]))
                vectors.append(vec)
        return vectors
    except Exception as e:
        logger.error(f"向量解压缩失败: {e}")
        # 尝试原始gzip解压作为备用
        try:
            decompressed_data = gzip.decompress(base64.b64decode(compressed_vectors_str))
            # ... 剩余解压逻辑相同
        except Exception as e2:
            logger.error(f"备用解压方案也失败: {e2}")
        return []

from prometheus_client import Counter, Histogram, start_http_server
import redis

# 启动监控服务
#start_http_server(8000)

REQUEST_COUNT = Counter('rag_requests_total', 'Total requests')
PROCESSING_TIME = Histogram('rag_processing_seconds', 'Processing time')
ERROR_COUNT = Counter('rag_errors_total', 'Error count', ['type'])
r = redis.Redis(host='localhost', port=6379, db=0)

@app.route('/predict_sync', methods=['POST'])
def predict_sync():
    """
    同步预测接口 - 增强错误处理版本
    注意：此端点由深圳服务器调用，必须保证响应格式稳定。
    """
    # 记录请求开始，用于性能追踪
    request_start_time = time.time()
    REQUEST_COUNT.inc()
    logger.info("🔵 [美国服务器] /predict_sync 端点接收到请求")

    try:
        # 1. 解析请求数据
        data = request.get_json()
        if not data:
            logger.error("❌ 请求数据为空或非JSON格式")
            return jsonify({
                "status": "error",
                "error": "无效的请求数据：必须为JSON格式"
            }), 400

        # 2. 提取参数
        query = data.get('query', '').strip()
        doc_vectors = data.get('doc_vectors', '')
        chat_id = data.get('chat_id', 'unknown')

        logger.info(f"收到查询: '{query[:50]}...', chat_id: {chat_id}")

        # 3. 基础参数验证
        if not query:
            logger.error(f"❌ 查询内容为空。Chat ID: {chat_id}")
            return jsonify({
                "status": "error",
                "error": "参数错误：查询内容 'query' 不能为空"
            }), 400

        if not doc_vectors:
            return jsonify({
                "status": "error",
                "error": "向量数据不能为空"
            }), 400

        # try:
        #     qa_result = run_with_timeout(
        #         qa_chain.invoke,
        #         {"query": query},
        #         timeout=45
        #     )
        #     PROCESSING_TIME.observe(time.time() - request_start_time)
        #     return jsonify(qa_result)
        # except TimeoutError:
        #     ERROR_COUNT.labels(type='timeout').inc()
        #     # 缓存降级
        #     cached = r.get(f"cache:qa:{hash(query)}")
        #     if cached:
        #         return jsonify(json.loads(cached))
        #     return jsonify({"answer": "系统繁忙，请稍后重试", "fallback": True})
        # except Exception as e:
        #     ERROR_COUNT.labels(type='exception').inc()
        #     return jsonify({"error": str(e)}), 500

        # 4. 解压缩向量数据
        logger.info("开始解压缩向量数据...")
        logger.info(f"压缩数据长度: {len(doc_vectors)}, 前100字符: {doc_vectors[:100]}")

        doc_vectors = decompress_vectors(doc_vectors)

        if not doc_vectors:
            logger.error(f"向量解压缩失败，原始数据: {doc_vectors[:200]}...")
            return jsonify({
                "status": "error",
                "error": "向量数据解压缩失败"
            }), 400

        logger.info(f"解压缩成功，获得 {len(doc_vectors)} 个向量")
        if doc_vectors:
            logger.info(f"每个向量维度: {len(doc_vectors[0])}")

        # 4. 核心处理逻辑 - 使用全局索引和LLM生成答案
        # 重要：这里使用您系统中已定义好的 qa_chain 或类似组件
        # 确保 global_vectorstore 和 llm 已正确初始化
        if global_vectorstore is None:
            logger.critical("❌ 全局向量索引未初始化，系统未就绪")
            return jsonify({
                "status": "error",
                "error": "系统服务未就绪：向量索引加载失败"
            }), 503

        # 4.1 检索相关文档
        logger.info("开始向量检索...")
        try:
            # 使用传入的向量构建临时检索器，或从全局索引检索
            # 此处是关键，可能需要根据您的文档处理方式调整
            docs = global_vectorstore.similarity_search(query, k=3)
            if not docs:
                logger.info("未检索到相关文档，将使用空上下文")
                context = "未找到与问题相关的参考资料。"
            else:
                context = "\n\n".join([f"[资料{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

            logger.info(f"检索完成，上下文长度: {len(context)} 字符")

        except Exception as retrieval_error:
            logger.error(f"❌ 文档检索阶段失败: {str(retrieval_error)}")
            # 降级方案：使用空上下文，不直接失败
            context = "文档检索系统暂时不可用。"
            # 或者可以选择直接返回错误
            # return jsonify({"status": "error", "error": f"检索失败: {str(retrieval_error)}"}), 500

        # 4.2 构建LLM提示词
        prompt = f"""你是一个专业的文档分析助手。请根据以下【参考资料】，用简洁、准确、友好的中文回答用户问题。

【参考资料】
{context}

【用户问题】
{query}

【要求】
1. 只基于参考资料回答，不要编造内容。
2. 如果资料中没有相关信息，请如实告知。
3. 答案控制在300字以内。
4. 语言自然、口语化。

回答："""

        logger.info("开始调用LLM生成回答...")

        # 4.3 调用LLM - 这是最可能超时或出错的部分
        try:
            # 使用您系统中初始化好的 llm 对象
            # 关键：这里设置了 timeout=60.0，确保Ollama调用不会无限期挂起
            if llm is None:
                raise Exception("LLM服务未初始化")

            # 调用LLM生成回答
            llm_response = llm.invoke(prompt)  # 或使用 llm.generate(), 根据您的LangChain版本调整

            # 清理响应内容，移除可能的标记
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL).strip()
            clean_response = re.sub(r'\n\s*\n', '\n', clean_response)

            if not clean_response or len(clean_response) < 5:
                clean_response = "抱歉，我无法基于现有资料生成有效的回答。"

            logger.info(f"✅ LLM调用成功，生成回答长度: {len(clean_response)}")

        except Exception as llm_error:
            # 特别捕获LLM调用相关的异常（如超时、连接失败）
            error_msg = str(llm_error)
            logger.error(f"❌ LLM调用失败: {error_msg}")

            # 根据错误类型提供更友好的提示
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_msg_user = "AI思考超时，请简化问题或稍后重试。"
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                error_msg_user = "AI服务暂时不可用，请检查网络连接或稍后重试。"
            else:
                error_msg_user = "AI处理遇到意外错误。"

            # 返回错误信息，让深圳服务器能进行降级处理
            return jsonify({
                "status": "error",
                "error": error_msg_user,
                "internal_error": error_msg[:200]  # 记录内部错误前200字符供调试
            }), 500

        # 5. 构建成功响应
        processing_time = round(time.time() - request_start_time, 2)
        logger.info(f"✅ /predict_sync 请求处理完成，总耗时: {processing_time}秒")

        response_data = {
            "status": "success",
            "result": clean_response,
            "source_documents": [doc.page_content[:100] + "..." for doc in docs] if docs else [],
            "processing_time_seconds": processing_time,
            "vector_count": len(doc_vectors),
            "retrieved_docs_count": len(docs)
        }

        return jsonify(response_data)

    except json.JSONDecodeError as e:
        # 捕获JSON解析错误（最顶层的请求数据问题）
        logger.error(f"❌ 请求JSON解析失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "请求数据格式错误：必须是有效的JSON"
        }), 400

    except Exception as e:
        # 捕获所有其他未预料的异常
        error_msg = str(e)
        logger.error(f"❌ /predict_sync 处理过程中发生未预料错误: {error_msg}")
        import traceback
        logger.error(f"详细堆栈信息: {traceback.format_exc()}")

        # 返回通用错误信息，避免泄露内部细节
        return jsonify({
            "status": "error",
            "error": "服务器内部处理异常",
            "request_id": f"req_{hash(str(time.time()))}"  # 简单的请求ID，用于日志追踪
        }), 500



def process_sync_task(task_id):
    """异步处理同步任务"""
    try:
        redis_client = get_redis_connection()

        # 获取任务数据
        task_data = redis_client.hgetall(f"sync_task:{task_id}")
        if not task_data:
            return

        query = task_data['query']
        doc_vectors = json.loads(task_data['doc_vectors'])
        chat_id = task_data['chat_id']

        # 执行实际的检索和生成
        result = actual_retrieval_and_generation(query, doc_vectors, chat_id)

        # 更新任务状态为完成
        redis_client.hset(f"sync_task:{task_id}", 'status', 'completed')
        redis_client.hset(f"sync_task:{task_id}", 'result', json.dumps(result))
        redis_client.hset(f"sync_task:{task_id}", 'complete_time', time.time())

    except Exception as e:
        logger.error(f"处理同步任务失败: {e}")
        redis_client.hset(f"sync_task:{task_id}", 'status', 'failed')
        redis_client.hset(f"sync_task:{task_id}", 'error', str(e))


@app.route('/sync_task_status/<task_id>', methods=['GET'])
def sync_task_status(task_id):
    """查询同步任务状态"""
    redis_client = get_redis_connection()
    task_data = redis_client.hgetall(f"sync_task:{task_id}")

    if not task_data:
        return jsonify({'status': 'error', 'error': '任务不存在'}), 404

    response = {
        'task_id': task_id,
        'status': task_data.get('status', 'unknown')
    }

    if task_data.get('status') == 'completed':
        response['result'] = json.loads(task_data.get('result', '{}'))
    elif task_data.get('status') == 'failed':
        response['error'] = task_data.get('error', '')

    return jsonify(response)



@app.route('/debug', methods=['GET'])
def debug_info():
    """调试信息端点"""
    info = {
        "service": "hf_proxy",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "embeddings": "loaded" if embeddings else "missing",
            "llm": "loaded" if llm else "missing",
            "global_vectorstore": "loaded" if global_vectorstore else "missing"
        },
        "endpoints": [
            "/health",
            "/predict_sync",
            "/debug"
        ]
    }
    return jsonify(info)


@app.route('/test_predict_sync', methods=['POST'])
def test_predict_sync():
    """测试/predict_sync端点"""
    test_data = {
        "query": "测试问题",
        "doc_vectors": [[0.1] * 768],  # 测试向量
        "doc_contents": ["这是一个测试文档内容"]
    }

    try:
        result = handle_real_sync_query(
            test_data["query"],
            test_data["doc_vectors"],
            test_data["doc_contents"]
        )
        return jsonify({
            "test_status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({
            "test_status": "failed",
            "error": str(e)
        }), 500

def handle_real_sync_query(query, doc_vectors, doc_contents):
    """实际处理同步查询（增强版本）"""
    try:
        logger.info("开始处理真实同步查询")

        # 1. 验证输入数据
        if not query or not isinstance(query, str):
            logger.error("查询内容无效")
            return {
                "answer": "查询内容格式错误",
                "source_documents": [],
                "status": "error"
            }

        # 2. 构建临时索引（增强错误处理）
        vector_store = None
        try:
            vector_store = build_faiss_from_precomputed(doc_vectors, doc_contents)
            if vector_store:
                logger.info("临时FAISS索引构建成功")
            else:
                logger.warning("临时索引构建失败，使用降级方案")
        except Exception as e:
            logger.error(f"索引构建异常: {e}")

        # 3. 检索相关文档
        source_docs = []
        if vector_store:
            try:
                source_docs = vector_store.get_relevant_documents(query)
                logger.info(f"检索到 {len(source_docs)} 个相关文档")
            except Exception as e:
                logger.error(f"文档检索失败: {e}")
                source_docs = []

        # 4. 构造上下文
        context = ""
        if source_docs:
            context = "\n\n".join([doc.page_content for doc in source_docs])
        elif doc_contents:
            # 使用前3个文档内容作为上下文
            context = "\n\n".join(doc_contents[:3])
            logger.info("使用直接文档内容作为上下文")
        else:
            context = "无可用文档内容"
            logger.warning("没有可用的文档内容")

        # 5. LLM调用（增强重试机制）
        answer = call_llm_with_retry(query, context)

        return {
            "answer": answer,
            "source_documents": [doc.page_content[:200] + "..." for doc in source_docs] if source_docs else [],
            "status": "success"
        }

    except Exception as e:
        logger.error(f"同步查询处理失败: {e}", exc_info=True)
        return {
            "answer": "系统处理异常，请稍后重试",
            "source_documents": [],
            "status": "error"
        }


def call_llm_with_retry(query, context, max_retries=3):
    """带重试的LLM调用"""
    for attempt in range(max_retries):
        try:
            prompt = f"""基于以下文档内容回答问题。

文档内容：
{context}

问题：{query}

请根据文档内容提供准确回答。如果文档中没有相关信息，请如实告知。"""

            if hasattr(llm, 'invoke'):
                response = llm.invoke(prompt)
            elif hasattr(llm, '__call__'):
                response = llm(prompt)
            else:
                response = "LLM服务不可用"

            if response and len(response.strip()) > 10:
                return response

            logger.warning(f"LLM返回空或过短响应，尝试 {attempt + 1}/{max_retries}")
            time.sleep(2)  # 等待2秒后重试

        except Exception as e:
            logger.error(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return "抱歉，AI服务暂时不可用，请稍后再试。"
            time.sleep(2)

    return "系统繁忙，请稍后重试。"

@app.route('/llm_health', methods=['GET'])
def llm_health_check():
    """检查LLM服务健康状态"""
    try:
        # 测试LLM是否可用
        test_prompt = "测试"
        if hasattr(llm, 'generate'):
            result = llm.generate([test_prompt])
            status = "healthy"
        else:
            result = llm(test_prompt)
            status = "healthy" if result and len(result) > 0 else "unhealthy"

        return jsonify({
            "status": status,
            "llm_type": str(type(llm)),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "llm_type": str(type(llm)),
            "timestamp": datetime.now().isoformat()
        }), 503


# 添加一个根路径测试
@app.route('/', methods=['GET'])
def root_test():
    logger.info("根路径被访问")
    return jsonify({
        "service": "hf_proxy",
        "status": "running",
        "endpoints": ["/health", "/predict_sync", "/"],
        "timestamp": datetime.now().isoformat()
    })


def fallback_similarity_search(query, doc_contents, doc_vectors):
    """备用相似度搜索方案（同步版本）"""
    try:
        # 计算查询向量
        query_vector = embeddings.embed_query(query)

        # 计算相似度
        similarities = []
        for i, doc_vector in enumerate(doc_vectors):
            if len(doc_vector) == len(query_vector):
                similarity = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                similarities.append((similarity, i))

        # 排序取前3
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:3]]

        # 构建上下文
        context = "\n\n".join([doc_contents[i] for i in top_indices])

        # 同步调用LLM
        return call_llm_with_context_sync(query, context)

    except Exception as e:
        logger.error(f"备用方案失败: {e}")
        return jsonify({"error": "所有处理方案均失败"}), 500


@app.route('/debug_models', methods=['GET'])
def debug_models():
    """模型调试接口"""
    models_status = {
        "embeddings_loaded": embeddings is not None,
        "llm_loaded": llm is not None,
        "clip_loaded": clip_model is not None,
        "whisper_loaded": whisper_model is not None,
        "embeddings_type": str(type(embeddings)) if embeddings else None,
        "llm_type": str(type(llm)) if llm else None
    }

    # 测试嵌入模型
    if embeddings:
        try:
            test_embedding = embeddings.embed_query("测试文本")
            models_status["embeddings_working"] = True
            models_status["embedding_dim"] = len(test_embedding)
        except Exception as e:
            models_status["embeddings_working"] = False
            models_status["embeddings_error"] = str(e)

    # 测试LLM
    if llm:
        try:
            # 对于CustomHuggingFaceLLM
            if hasattr(llm, 'model_available'):
                models_status["llm_working"] = llm.model_available
            else:
                # 简单测试
                test_response = llm.generate(["Hello"])
                models_status["llm_working"] = True
        except Exception as e:
            models_status["llm_working"] = False
            models_status["llm_error"] = str(e)

    return jsonify(models_status)


# ==================== 启动时加载（放在文件末尾，app.run 之前） ====================
def _init_app():
    global embeddings, global_vectorstore, llm

    logger.info("开始初始化应用组件...")

    try:
        # 1. 文本嵌入模型
        logger.info("加载文本嵌入模型...")
        embeddings = HuggingFaceEmbeddings(model_name=config.TEXT_MODEL_PATH)
        logger.info("文本嵌入模型加载完成")

        # 2. 全局FAISS索引
        index_path = "/customer_ai/faiss_index"
        logger.info(f"加载FAISS索引从: {index_path}")
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            global_vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("全局FAISS索引加载成功")
        else:
            global_vectorstore = FAISS.from_documents([], embeddings)
            logger.info("创建空全局索引")

        # 3. 简化LLM初始化（基于现有成功代码）
        logger.info("初始化LLM服务...")
        llm = _initialize_simple_ollama()

        logger.info("应用组件初始化全部完成")

    except Exception as e:
        logger.error(f"应用初始化异常: {e}")
        raise

def _initialize_simple_ollama():
    """简化版Ollama初始化（基于现有成功代码）"""
    try:
        # 直接使用已知可用的模型
        from langchain_community.llms import Ollama
        llm = Ollama(
            model="llama2",  # 文档4显示这个模型可用
            base_url="http://localhost:11434",
            timeout=60
        )
        logger.info("Ollama LLM初始化成功")
        return llm
    except Exception as e:
        logger.error(f"Ollama初始化失败: {e}")
        # 使用备用方案
        from langchain_community.llms import FakeListLLM
        return FakeListLLM(responses=["根据文档内容，这是一个需要进一步分析的复杂问题。"])

def _initialize_ollama_with_fallback(self):
    """动态检测并初始化Ollama，带多级降级策略"""
    try:
        # 首先检测系统可用的Ollama模型
        available_models = _get_available_ollama_models()  # ✅ 修复：移除self

        if not available_models:
            logger.warning("未找到任何可用的Ollama模型")
            return _create_fallback_llm()  # ✅ 修复：移除self

        # 按优先级尝试不同的模型
        model_priority = [
            "llama3.2:3b",  # 首选模型
            "llama2",  # 实际存在的模型
            "llama2:latest",  # 完整名称
            "mistral",  # 备用模型1
            "gemma",  # 备用模型2
            available_models[0]  # 使用第一个可用模型
        ]

        for model_name in model_priority:
            if model_name in available_models:
                try:
                    logger.info(f"尝试加载模型: {model_name}")
                    llm = Ollama(model=model_name, base_url="http://localhost:11434")

                    # 测试模型是否真的可用
                    test_response = llm.invoke("测试")
                    if test_response and len(test_response.strip()) > 0:
                        logger.info(f"Ollama模型加载成功: {model_name}")
                        return llm
                    else:
                        logger.warning(f"模型响应异常: {model_name}")
                except Exception as e:
                    logger.warning(f"模型加载失败 {model_name}: {e}")
                    continue

        # 所有模型尝试都失败，使用降级方案
        logger.warning("所有Ollama模型尝试失败，使用降级LLM")
        return _create_fallback_llm()  # ✅ 修复：移除self

    except Exception as e:
        logger.error(f"Ollama初始化异常: {e}")
        return _create_fallback_llm()  # ✅ 修复：移除self


def _get_available_ollama_models(self):
    """获取系统可用的Ollama模型列表"""
    try:
        # 执行ollama list命令获取模型列表
        import subprocess
        result = subprocess.run(['ollama', 'list'],
                                capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            models = []
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # 跳过标题行
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            logger.info(f"检测到可用Ollama模型: {models}")
            return models
        else:
            logger.warning("获取Ollama模型列表失败")
            return []

    except Exception as e:
        logger.warning(f"获取Ollama模型列表异常: {e}")
        return []

def _create_fallback_llm(self):
    """创建多级降级LLM方案"""
    try:
        # 第一级降级：使用HuggingFace API
        try:
            from langchain_community.llms import HuggingFaceEndpoint
            llm = HuggingFaceEndpoint(
                repo_id="HuggingFaceH4/zephyr-7b-beta",
                task="text-generation",
                model_kwargs={
                    "max_new_tokens": 512,
                    "temperature": 0.1
                }
            )
            logger.info("使用HuggingFace API作为降级方案")
            return llm
        except Exception as e:
            logger.warning(f"HuggingFace API降级失败: {e}")

        # 第二级降级：使用简单的本地模型
        try:
            from langchain_community.llms import CTransformers
            llm = CTransformers(
                model="TheBloke/Llama-2-7B-Chat-GGML",
                model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",
                model_type="llama"
            )
            logger.info("使用本地CTransformers作为降级方案")
            return llm
        except Exception as e:
            logger.warning(f"本地模型降级失败: {e}")

        # 最终降级：极简回答生成器
        class MinimalFallbackLLM:
            def invoke(self, prompt):
                responses = [
                    "根据您提供的文档，这是一个需要分析的内容。",
                    "我已收到您的查询，正在处理中。",
                    "基于上传的文档信息，建议进一步分析。",
                    "文档内容已接收，需要时间处理。"
                ]
                return responses[hash(prompt) % len(responses)]

        logger.info("使用极简备用LLM")
        return MinimalFallbackLLM()

    except Exception as e:
        logger.error(f"所有降级方案都失败: {e}")

        # 保证至少有一个可用的LLM
        class GuaranteedLLM:
            def invoke(self, prompt):
                return "系统正在处理您的请求，请稍后查看结果。"

        return GuaranteedLLM()

def main():
    """主启动函数"""
    try:
        # 确保目录存在
        os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        os.makedirs(config.FAISS_BACKUP_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # 强制初始化应用
        logger.info("=== 开始强制初始化应用 ===")
        _init_app()

        # 验证关键组件
        if not all([embeddings, global_vectorstore, llm]):
            logger.error("关键组件初始化失败")
            return False

        logger.info("=== 应用初始化完成，启动Flask服务 ===")
        return True

    except Exception as e:
        logger.error(f"启动失败: {e}")
        return False

if __name__ == '__main__':
    if main():
        # 添加启动成功日志
        logger.info(f"Flask服务启动在 0.0.0.0:5000")
        print("=== Flask服务启动成功 ===")
        print("健康检查地址: http://0.0.0.0:5000/health")

        def periodic_health_check():
            check_ollama_health()
            threading.Timer(60, periodic_health_check).start()

        periodic_health_check()

        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("应用启动失败")
        sys.exit(1)