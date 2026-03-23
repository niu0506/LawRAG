"""
配置文件

定义系统配置参数，包括：
- LLM配置（支持OpenAI兼容格式的API）
- 向量存储配置
- Embedding模型配置
- 应用服务器配置

使用Pydantic Settings进行环境变量管理和类型验证。
"""

import os
from typing import List, Dict, Any

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置类
    
    使用Pydantic Settings从环境变量或.env文件加载配置。
    所有配置项都有默认值，可以在环境变量中覆盖。
    
    配置项说明：
    - LLM相关: API密钥、Base URL、模型名称
    - Embedding: HuggingFace模型配置
    - RAG: 文本分块、检索参数
    - 存储: 向量数据库路径、文档目录
    - 服务器: 主机、端口、CORS配置
    """
    
    # ==================== LLM配置 ====================
    # OpenAI兼容API的密钥
    LLM_API_KEY: SecretStr = SecretStr("")
    # OpenAI兼容API的Base URL
    LLM_BASE_URL: str = ""
    # LLM模型名称
    LLM_MODEL: str = ""
    
    # ==================== Embedding模型配置 ====================
    # HuggingFace访问令牌（用于私有模型）
    HF_TOKEN: str = ""
    # 中文Embedding模型，使用BAAI/bge-large-zh-v1.5
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    
    # ==================== RAG引擎配置 ====================
    # 文本分割块大小（字符数）
    CHUNK_SIZE: int = 500
    # 文本分割重叠区域大小
    CHUNK_OVERLAP: int = 50
    # 检索返回的Top-K结果数量
    TOP_K: int = 5
    # 对话历史保留的轮次
    HISTORY_TURNS: int = 5
    
    # ==================== 存储路径配置 ====================
    # Chroma向量数据库存储路径
    CHROMA_DB_PATH: str = "./db/chroma"
    # 法律文档存放目录
    LAWS_DIR: str = "./data"
    # 已处理文件的MD5哈希缓存文件路径
    FILE_HASH_CACHE: str = "./db/processed_files.json"
    
    # ==================== 服务器配置 ====================
    # 服务器监听地址
    HOST: str = "localhost"
    # 服务器监听端口
    PORT: int = 8000
    # CORS允许的来源列表，"*"表示允许所有
    CORS_ORIGINS: List[str] = ["*"]
    # 文件上传大小限制，默认50MB
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")


settings = Settings()

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN


def get_llm():
    """
    获取LLM实例
    
    Returns:
        LangChain ChatOpenAI实例
    
    Raises:
        ValueError: 当API Key未配置时
    """
    from langchain_openai import ChatOpenAI
    
    if not settings.LLM_API_KEY.get_secret_value():
        raise ValueError("未配置 LLM_API_KEY")
    
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL
    )


def get_llm_info() -> Dict[str, Any]:
    """
    获取当前LLM配置信息
    
    Returns:
        包含name和model的字典
    """
    model_name = settings.LLM_MODEL
    if "/" in model_name:
        name = model_name.split("/")[-1]
    elif ":" in model_name:
        name = model_name.split(":")[0]
    else:
        name = model_name
    return {
        "name": name,
        "model": model_name,
    }