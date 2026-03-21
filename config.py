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
import warnings
from typing import List, Dict, Any
from pydantic import SecretStr

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", message=".*unexpected.*")

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    应用配置类
    
    使用Pydantic Settings从环境变量或.env文件加载配置。
    所有配置项都有默认值，可以在环境变量中覆盖。
    """
    
    # ==================== LLM配置 ====================
    LLM_API_KEY: SecretStr = SecretStr("")
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""
    
    # ==================== Embedding模型配置 ====================
    HF_TOKEN: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    
    # ==================== RAG引擎配置 ====================
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    
    # ==================== 存储路径配置 ====================
    CHROMA_DB_PATH: str = "./db/chroma"
    LAWS_DIR: str = "./data"
    HISTORY_DB_PATH: str = "./db/history.db"
    FILE_HASH_CACHE: str = "./db/processed_files.json"
    
    # ==================== 服务器配置 ====================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024

    model_config = SettingsConfigDict(extra="ignore")


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
        包含model的字典
    """
    return {
        "model": settings.LLM_MODEL,
    }