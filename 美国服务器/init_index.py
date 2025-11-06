from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import logging
import shutil
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndexVersionManager:
    """索引版本管理类"""
    
    def __init__(self, base_dir="/app/faiss_index"):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.metadata_file = self.base_dir / "version_metadata.json"
        self.max_versions = 5  # 保留的最大版本数
        
        # 创建必要目录
        self.versions_dir.mkdir(parents=True, exist_ok=True)
    
    def get_current_version(self):
        """获取当前版本信息"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_version_metadata(self, version_info):
        """保存版本元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(version_info, f, indent=2)
    
    def create_new_version(self, index_data):
        """创建新版本"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        version_id = f"v{timestamp}_{version_hash}"
        version_dir = self.versions_dir / version_id
        
        # 创建版本目录
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        index_data.save_local(str(version_dir))
        
        # 更新版本信息
        version_info = {
            "current_version": version_id,
            "created_at": datetime.now().isoformat(),
            "versions": self.get_current_version().get("versions", [])
        }
        
        # 添加新版本到版本列表
        version_info["versions"].insert(0, {
            "version_id": version_id,
            "created_at": version_info["created_at"],
            "path": str(version_dir)
        })
        
        # 限制版本数量
        version_info["versions"] = version_info["versions"][:self.max_versions]
        
        self.save_version_metadata(version_info)
        
        # 更新当前版本符号链接
        current_link = self.base_dir / "current"
        if current_link.exists():
            current_link.unlink()
        current_link.symlink_to(version_dir, target_is_directory=True)
        
        return version_id
    
    def cleanup_old_versions(self):
        """清理旧版本"""
        version_info = self.get_current_version()
        if not version_info:
            return
        
        current_versions = version_info.get("versions", [])
        if len(current_versions) <= self.max_versions:
            return
        
        # 删除超出的旧版本
        for version_data in current_versions[self.max_versions:]:
            try:
                version_path = Path(version_data["path"])
                if version_path.exists():
                    shutil.rmtree(version_path)
                    logger.info(f"清理旧版本: {version_data['version_id']}")
            except Exception as e:
                logger.warning(f"清理版本失败 {version_data['version_id']}: {e}")
        
        # 更新元数据
        version_info["versions"] = current_versions[:self.max_versions]
        self.save_version_metadata(version_info)

class BackupManager:
    """备份管理类"""
    
    def __init__(self, backup_base="/app/faiss_index_backup"):
        self.backup_base = Path(backup_base)
        self.backup_base.mkdir(parents=True, exist_ok=True)
    
    def create_incremental_backup(self, source_dir, version_id):
        """创建增量备份"""
        backup_dir = self.backup_base / f"backup_{version_id}"
        
        try:
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            shutil.copytree(source_dir, backup_dir)
            logger.info(f"创建增量备份: {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count=3):
        """清理旧备份"""
        try:
            backups = sorted(self.backup_base.glob("backup_*"), key=os.path.getmtime)
            if len(backups) <= keep_count:
                return
            
            for old_backup in backups[:-keep_count]:
                shutil.rmtree(old_backup)
                logger.info(f"清理旧备份: {old_backup}")
        except Exception as e:
            logger.warning(f"清理备份失败: {e}")

def validate_index_integrity(index_path):
    """验证索引完整性"""
    try:
        # 检查必要文件是否存在
        required_files = ["index.faiss", "index.pkl"]
        for file in required_files:
            if not (Path(index_path) / file).exists():
                return False, f"缺少必要文件: {file}"
        
        # 尝试加载索引验证完整性
        embeddings = HuggingFaceEmbeddings(
            model_name="/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e",
            model_kwargs={'device': 'cpu'},
            cache_folder="/app/models"
        )
        
        vectorstore = FAISS.load_local(str(index_path), embeddings)
        
        # 检查索引基本信息
        if hasattr(vectorstore, 'index') and vectorstore.index is not None:
            return True, "索引完整性验证通过"
        else:
            return False, "索引加载失败"
            
    except Exception as e:
        return False, f"索引验证异常: {str(e)}"

def init_faiss_index(max_retries=3):
    """初始化FAISS索引（带重试机制和版本管理）"""
    
    # 初始化管理器
    version_manager = IndexVersionManager()
    backup_manager = BackupManager()
    
    for attempt in range(max_retries):
        try:
            # 加载嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="/root/.cache/huggingface/hub/models--shibing624--text2vec-base-chinese/snapshots/183bb99aa7af74355fb58d16edf8c13ae7c5433e",
                model_kwargs={'device': 'cpu'},
                cache_folder="/app/models"
            )
            
            # 检查当前版本是否存在且有效
            current_version = version_manager.get_current_version().get("current_version")
            if current_version:
                current_path = version_manager.versions_dir / current_version
                is_valid, message = validate_index_integrity(current_path)
                if is_valid:
                    logger.info("发现有效的现有索引，跳过初始化")
                    return True
                else:
                    logger.warning(f"现有索引无效: {message}，将重新初始化")
            
            # 初始化空索引
            vectorstore = FAISS.from_texts(["系统初始化文档"], embeddings)
            
            # 创建新版本
            version_id = version_manager.create_new_version(vectorstore)
            
            # 验证新索引
            version_path = version_manager.versions_dir / version_id
            is_valid, message = validate_index_integrity(version_path)
            if not is_valid:
                raise Exception(f"新索引验证失败: {message}")
            
            # 创建备份
            backup_manager.create_incremental_backup(version_path, version_id)
            
            # 清理旧版本和备份
            version_manager.cleanup_old_versions()
            backup_manager.cleanup_old_backups()
            
            logger.info(f"FAISS索引初始化完成，版本: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"索引初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return False
            time.sleep(2)
    
    return False

if __name__ == '__main__':
    success = init_faiss_index()
    exit(0 if success else 1)