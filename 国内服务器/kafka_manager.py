#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic

class KafkaTopicManager:
    def __init__(self, bootstrap_servers='39.108.11.214:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.admin_client = KafkaAdminClient(
            bootstrap_servers=[bootstrap_servers],
            client_id='rag_topic_manager'
        )
    
    def create_topic(self, topic_name, num_partitions=1, replication_factor=1):
        """创建主题"""
        try:
            topic_list = [NewTopic(
                name=topic_name,
                num_partitions=num_partitions,
                replication_factor=replication_factor
            )]
            self.admin_client.create_topics(new_topics=topic_list, validate_only=False)
            print(f"主题创建成功: {topic_name}")
            return True
        except Exception as e:
            print(f"主题创建失败: {e}")
            return False
    
    def list_topics(self):
        """列出所有主题"""
        try:
            topics = self.admin_client.list_topics()
            print("现有主题:")
            for topic in topics:
                print(f"  - {topic}")
            return topics
        except Exception as e:
            print(f"获取主题列表失败: {e}")
            return []
    
    def test_connection(self):
        """测试Kafka连接"""
        try:
            producer = KafkaProducer(bootstrap_servers=[self.bootstrap_servers])
            producer.close()
            print("Kafka连接测试成功")
            return True
        except Exception as e:
            print(f"Kafka连接测试失败: {e}")
            return False

if __name__ == "__main__":
    manager = KafkaTopicManager()
    
    # 测试连接
    if manager.test_connection():
        # 列出主题
        manager.list_topics()
        
        # 创建必要的主题
        required_topics = ['vector-data', 'rag.file.upload.shenzhen', 'rag.task.status.shenzhen']
        for topic in required_topics:
            manager.create_topic(topic)