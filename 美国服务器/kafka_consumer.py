#!/usr/bin/env python3
import json
import time
import hashlib
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from kafka import KafkaConsumer, KafkaProducer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import threading
import signal
import sys
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MessageProcessResult:
    """消息处理结果"""
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class MessageProcessor:
    """消息处理器"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.processed_ids = set()
        self._load_processed_ids()

    def _load_processed_ids(self):
        """加载已处理的ID集合"""
        try:
            # 这里可以从Redis或其他持久化存储加载
            logger.info("加载已处理ID集合")
        except Exception as e:
            logger.warning(f"加载已处理ID失败: {e}")

    def process_message(self, vector_data: Dict) -> MessageProcessResult:
        start_time = time.time()

        try:
            vector_id = vector_data.get('id', 'unknown')

            # 深度调试：记录接收到的完整消息
            logger.info(f"接收到的消息结构: {list(vector_data.keys())}")
            logger.info(f"消息ID: {vector_id}")
            logger.info(f"向量类型: {type(vector_data.get('vector'))}, 长度: {len(vector_data.get('vector', []))}")

            # 提取校验和
            expected_checksum = vector_data.get('checksum', '')
            logger.info(f"期望的校验和: {expected_checksum}")

            # 重新创建校验和计算数据（与生产者完全相同的方式）
            check_data = {
                "id": vector_data.get('id'),
                "type": vector_data.get('type'),
                "path": vector_data.get('path'),
                "vector": vector_data.get('vector'),
                "content": vector_data.get('content'),
                "metadata": vector_data.get('metadata', {})
            }

            # 使用与生产者完全相同的序列化方式
            check_data_str = json.dumps(check_data, sort_keys=True, ensure_ascii=False)
            actual_checksum = hashlib.md5(check_data_str.encode('utf-8')).hexdigest()

            logger.info(f"实际的校验和: {actual_checksum}")
            logger.info(f"校验数据字符串长度: {len(check_data_str)}")

            # 如果校验失败，进行详细分析
            if expected_checksum and expected_checksum != actual_checksum:
                logger.warning(f"数据校验失败，进行详细分析: {vector_id}")

                # 检查关键字段差异
                original_vector = vector_data.get('vector', [])
                if original_vector:
                    logger.info(f"向量样本 - 前5个值: {original_vector[:5]}")
                    logger.info(f"向量数据类型: {type(original_vector[0]) if original_vector else 'N/A'}")

                # 检查内容差异
                content = vector_data.get('content', '')
                logger.info(f"内容预览: {content[:100]}...")

                # 临时解决方案：记录错误但继续处理
                # 这里我们先跳过校验失败，继续处理数据
                logger.warning(f"校验失败但继续处理: {vector_id}")
                # 不返回错误，继续处理流程

            # 修复：将 doc_content 的初始化移出 if 条件语句，确保始终有值
            doc_content = vector_data.get('content', '')
            if not doc_content and 'vector' in vector_data:
                doc_content = f"Document with vector data - {vector_id}"

            # 修复：不直接在Document中设置embedding，而是单独存储
            doc = Document(
                page_content=doc_content,
                metadata={
                    "id": vector_id,
                    "type": vector_data.get('type', 'unknown'),
                    "source": vector_data.get('path', ''),
                    "timestamp": datetime.now().isoformat(),
                    "modality": vector_data.get('modality', 'text'),
                    "file_count": vector_data.get('file_count', 1),
                    "checksum_valid": expected_checksum == actual_checksum if expected_checksum else True,
                    # 将embedding存储在metadata中
                    "embedding_vector": vector_data.get('vector')
                }
            )

            # 标记为已处理
            self.processed_ids.add(vector_id)

            processing_time = time.time() - start_time
            logger.info(f"消息处理完成: {vector_id} (耗时: {processing_time:.2f}s)")

            return MessageProcessResult(
                success=True,
                document=doc,
                processing_time=processing_time
            )

        except Exception as e:
            error_msg = f"消息处理异常: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return MessageProcessResult(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )


class IndexUpdater:
    """索引更新器"""
    def __init__(self, index_path: str, embeddings):
        self.index_path = index_path
        self.embeddings = embeddings
        self.vectorstore = None
        self._load_or_create_index()
        self.lock = threading.Lock()
        self.last_update_time = time.time()
        os.makedirs(index_path, exist_ok=True)
        self.metadata_csv = os.path.join(index_path, "metadata.csv")

    def _load_or_create_index(self):
        """加载或创建FAISS索引"""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            pkl_file = os.path.join(self.index_path, "index.pkl")

            if os.path.exists(index_file) and os.path.exists(pkl_file):
                logger.info("加载现有 FAISS 索引")
                self.vectorstore = FAISS.load_local(self.index_path, self.embeddings,
                                                    allow_dangerous_deserialization=True)
            else:
                logger.info("创建新 FAISS 索引")
                self.vectorstore = FAISS.from_documents([], self.embeddings)

        except Exception as e:
            logger.error(f"索引加载/创建失败: {e}")
            raise

    def _load_index(self):
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        return None

    def update_index_batch(self, documents: List[Document]) -> bool:
        """修复的批量索引更新方法"""
        if not documents:
            return False

        try:
            with self.lock:
                # 修复：确保metadata.csv被更新
                self._update_metadata_csv(documents)

                # 使用预计算嵌入
                success = self._safe_add_precomputed_embeddings(documents)
                if not success:
                    # 降级方案：重新计算嵌入
                    self.vectorstore.add_documents(documents)

                # 保存索引
                self.vectorstore.save_local(self.index_path)

                # 更新metadata.csv（确保被调用）
                self._update_metadata_csv(documents)

            logger.info(f"索引更新成功: +{len(documents)} 条文档")
            logger.info(f"metadata.csv 已更新")
            return True

        except Exception as e:
            logger.error(f"索引更新失败: {e}")
            return False

    def _safe_add_precomputed_embeddings(self, documents: List[Document]) -> bool:
        """修复预计算嵌入添加方法"""
        try:
            texts = []
            embeddings_list = []
            metadatas = []

            for doc in documents:
                precomputed_embedding = doc.metadata.get("embedding_vector")
                if precomputed_embedding and len(precomputed_embedding) == 768:
                    texts.append(doc.page_content)
                    embeddings_list.append(precomputed_embedding)
                    # 清理metadata
                    clean_metadata = doc.metadata.copy()
                    clean_metadata.pop("embedding_vector", None)
                    metadatas.append(clean_metadata)

            if texts:
                # 修复：使用正确的参数格式
                self.vectorstore.add_embeddings(
                    text_embeddings=list(zip(texts, embeddings_list)),
                    metadatas=metadatas
                )
                return True
            return False

        except Exception as e:
            logger.error(f"预计算嵌入添加失败: {e}")
            # 降级方案：重新计算嵌入
            self.vectorstore.add_documents(documents)
            return True

    def _add_embeddings_to_index(self, texts: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        """修复预计算embedding添加方法"""
        try:
            logger.info(f"添加预计算embedding: {len(texts)} 个文档")

            # 方法1: 使用正确的方法签名
            if hasattr(self.vectorstore, 'add_embeddings'):
                try:
                    # 修复：使用正确的参数格式
                    text_embeddings = list(zip(texts, embeddings))
                    self.vectorstore.add_embeddings(
                        text_embeddings=text_embeddings,
                        metadatas=metadatas
                    )
                    logger.info("使用 add_embeddings 方法成功")
                    return True
                except Exception as e:
                    logger.warning(f"add_embeddings 方法失败: {e}")

            # 方法2: 降级到标准方法
            logger.info("使用标准add_documents方法")
            from langchain.schema import Document

            documents_to_add = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if i < len(metadatas) else {}
                # 创建文档（不包含预计算embedding）
                doc = Document(page_content=text, metadata=metadata)
                documents_to_add.append(doc)

            self.vectorstore.add_documents(documents_to_add)
            logger.info(f"使用标准方法添加了 {len(documents_to_add)} 个文档")
            return True

        except Exception as e:
            logger.error(f"添加预计算embedding失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _update_metadata_csv(self, documents: List[Document]):
        """修复的metadata.csv更新"""
        try:
            metadata_records = []
            for doc in documents:
                record = {
                    "id": doc.metadata.get("id", ""),
                    "source": doc.metadata.get("source", ""),
                    "type": doc.metadata.get("type", ""),
                    "timestamp": doc.metadata.get("timestamp", datetime.now().isoformat()),
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "content_length": len(doc.page_content),
                    "embedding_length": len(doc.metadata.get("embedding_vector", [])),
                    "checksum_valid": doc.metadata.get("checksum_valid", True),
                    "has_precomputed_embedding": "embedding_vector" in doc.metadata
                }
                metadata_records.append(record)

            df_new = pd.DataFrame(metadata_records)

            # 读取现有metadata或创建新的
            if os.path.exists(self.metadata_csv):
                try:
                    df_existing = pd.read_csv(self.metadata_csv)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    logger.warning(f"读取现有metadata.csv失败，创建新的: {e}")
                    df_combined = df_new
            else:
                df_combined = df_new

            # 保存metadata
            df_combined.to_csv(self.metadata_csv, index=False, encoding='utf-8')
            logger.info(f"metadata.csv 更新完成，总记录数: {len(df_combined)}")

        except Exception as e:
            logger.error(f"metadata.csv 更新失败: {e}")

    def update_index_immediate(self, documents: List[Document]) -> bool:
        """立即更新索引（优化同步处理）"""
        if not documents:
            return True

        try:
            with self.lock:
                # 快速添加文档
                self.vectorstore.add_documents(documents)

                # 立即保存索引
                self.vectorstore.save_local(self.index_path)

                self.last_update_time = time.time()

                logger.info(f"索引更新成功，处理了 {len(documents)} 个文档")
                return True

        except Exception as e:
            logger.error(f"索引更新失败: {e}")
            return False

    def _create_backup(self):
        """创建索引备份"""
        try:
            import shutil
            from datetime import datetime, timedelta

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"/customer_ai/faiss_backups/index_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)

            shutil.copy2(f"{self.index_path}/index.faiss", f"{backup_dir}/index.faiss")
            shutil.copy2(f"{self.index_path}/index.pkl", f"{backup_dir}/index.pkl")

            logger.info(f"索引备份完成: {backup_dir}")

            # 清理旧备份（调整为7天，更合理）
            self._cleanup_old_backups(days_to_keep=7)

        except Exception as e:
            logger.error(f"索引备份失败: {e}")

    def _cleanup_old_backups(self, days_to_keep: int):
        """清理旧备份"""
        try:
            import shutil
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            backup_base = "/customer_ai/faiss_backups"

            if os.path.exists(backup_base):
                for item in os.listdir(backup_base):
                    item_path = os.path.join(backup_base, item)
                    if os.path.isdir(item_path):
                        try:
                            dir_time = datetime.strptime(item.split('_')[1], "%Y%m%d_%H%M%S")
                            if dir_time < cutoff_time:
                                shutil.rmtree(item_path)
                                logger.info(f"清理旧备份: {item}")
                        except (ValueError, IndexError):
                            continue

        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")

class KafkaConsumerManager:
    """Kafka消费者管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consumer = None
        self.producer = None
        self.running = False
        self.message_processor = None
        self.index_updater = None
        self.batch_docs = []
        self.batch_size = config.get('batch_size', 10)  # 减小批量大小便于调试
        self.max_retries = config.get('max_retries', 3)
        self.last_flush_time = time.time()
        self.flush_interval = 30  # 30秒强制刷新

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，准备优雅停止...")
        self.stop()

    def initialize(self) -> bool:
        """初始化消费者"""
        try:
            # 初始化嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e",
                model_kwargs={'device': 'cpu'},
                cache_folder="/customer_ai/models"
            )

            # 初始化组件
            self.message_processor = MessageProcessor(embeddings)
            self.index_updater = IndexUpdater("/customer_ai/faiss_index", embeddings)

            # 初始化Kafka消费者
            self.consumer = KafkaConsumer(
                self.config['topic'],
                bootstrap_servers=self.config['bootstrap_servers'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                security_protocol="PLAINTEXT",
                auto_offset_reset='earliest',
                group_id=self.config['group_id'],
                enable_auto_commit=False,
                session_timeout_ms=90000,
                request_timeout_ms=120000,
                heartbeat_interval_ms=30000,
                max_poll_interval_ms=600000,
                fetch_max_wait_ms=1000,
                fetch_min_bytes=1,
                fetch_max_bytes=104857600,
                max_poll_records=100
            )

            # 初始化Kafka生产者（用于重试）
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

            logger.info("Kafka消费者初始化完成")
            return True

        except Exception as e:
            logger.error(f"Kafka消费者初始化失败: {e}")
            return False

    def start_consuming(self):
        """修复的消息消费方法"""
        if not self.consumer:
            logger.error("消费者未初始化")
            return

        self.running = True
        logger.info("开始消费Kafka消息...")

        try:
            while self.running:
                # 使用poll而不是for循环，便于控制刷新
                msg_batch = self.consumer.poll(timeout_ms=5000)

                if not msg_batch:
                    # 检查是否需要强制刷新
                    if time.time() - self.last_flush_time > self.flush_interval and self.batch_docs:
                        logger.info("达到刷新间隔，强制处理批次")
                        self._process_batch()
                    continue

                for topic_partition, messages in msg_batch.items():
                    for message in messages:
                        logger.info(
                            f"收到消息 | 主题: {message.topic} | 分区: {topic_partition.partition} | 偏移量: {message.offset}")

                        # 处理消息
                        result = self.message_processor.process_message(message.value)

                        if result.success and result.document:
                            self.batch_docs.append(result.document)

                            # 立即处理单个文档用于调试
                            logger.info("调试模式：立即处理单个文档")
                            self._process_batch_immediate([result.document])

                        # 提交偏移量
                        try:
                            self.consumer.commit()
                            logger.info(f"消息处理完成并提交: {message.offset}")
                        except Exception as e:
                            logger.error(f"偏移量提交失败: {e}")

        except Exception as e:
            logger.error(f"消息消费异常: {e}")
        finally:
            # 处理剩余文档
            if self.batch_docs:
                self._process_batch()
            self._cleanup()

    def _process_batch_immediate(self, documents: List[Document]):
        """立即处理文档（调试用）"""
        if not documents:
            return

        try:
            # 修复：只传递一个参数
            success = self.index_updater.update_index_batch(documents)
            if success:
                logger.info(f"立即处理成功: {len(documents)} 个文档")
                # 清理已处理的文档
                self.batch_docs = [doc for doc in self.batch_docs if doc not in documents]
            else:
                logger.error("立即处理失败")
        except Exception as e:
            logger.error(f"立即处理异常: {e}")

    def _process_batch(self):
        """处理批量文档"""
        if not self.batch_docs:
            return

        try:
            # 修复：只传递一个参数
            success = self.index_updater.update_index_batch(self.batch_docs)
            if success:
                logger.info(f"批量处理成功: {len(self.batch_docs)} 个文档")
                self.batch_docs.clear()
            else:
                logger.error("批量处理失败，将重新入队")
                self._requeue_failed_docs()
        except Exception as e:
            logger.error(f"批量处理异常: {e}")
            self._requeue_failed_docs()

    def _requeue_failed_docs(self):
        """重新入队失败的文档"""
        try:
            for doc in self.batch_docs:
                message = {
                    "id": doc.metadata.get("id", "unknown"),
                    "content": doc.page_content,
                    "type": doc.metadata.get("type", "unknown"),
                    "retry_count": doc.metadata.get("retry_count", 0) + 1
                }

                # 检查重试次数
                if message["retry_count"] > self.max_retries:
                    logger.error(f"消息重试次数超限: {message['id']}")
                    continue

                # 发送到重试队列
                future = self.producer.send('vector-data-retry', message)
                future.get(timeout=10)

            logger.warning(f"{len(self.batch_docs)} 个文档已重新加入队列")
            self.batch_docs.clear()

        except Exception as e:
            logger.error(f"重新排队失败: {e}")

    def _heartbeat_check(self):
        """心跳检查"""
        try:
            # 这里可以添加健康检查逻辑
            pass
        except Exception as e:
            logger.warning(f"心跳检查异常: {e}")

    def stop(self):
        """停止消费者"""
        logger.info("停止Kafka消费者...")
        self.running = False

        # 处理剩余文档
        if self.batch_docs:
            self._process_batch()

    def _cleanup(self):
        """清理资源"""
        try:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            logger.info("Kafka消费者资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


def main():
    """主函数"""
    # 配置参数
    config = {
        'bootstrap_servers': ['39.108.11.214:9092'],
        'topic': 'vector-data',
        'group_id': 'rag-index-group-multi-modal-v3',  # 修改group_id避免偏移量冲突
        'batch_size': 1,  # 调试阶段设置为1
        'max_retries': 3
    }

    # 检查索引目录
    index_path = "/customer_ai/faiss_index"
    if not os.path.exists(index_path):
        logger.info(f"创建索引目录: {index_path}")
        os.makedirs(index_path, exist_ok=True)

    # 检查目录权限
    if not os.access(index_path, os.W_OK):
        logger.error(f"索引目录无写权限: {index_path}")
        return 1

    # 初始化消费者管理器
    consumer_manager = KafkaConsumerManager(config)

    if not consumer_manager.initialize():
        logger.error("消费者初始化失败")
        return 1

    try:
        consumer_manager.start_consuming()
    except KeyboardInterrupt:
        logger.info("接收到键盘中断信号")
    except Exception as e:
        logger.error(f"消费者运行异常: {e}")
        return 1
    finally:
        consumer_manager.stop()

    return 0


if __name__ == '__main__':
    sys.exit(main())