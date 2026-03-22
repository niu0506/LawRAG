"""
RAG 引擎 — 向量检索 + LLM 生成 + 文档加载

本模块实现了AI法律顾问的核心RAG（检索增强生成）引擎，功能包括：

1. 文档加载与处理：
   - 支持多种格式（PDF、Word、PowerPoint、Excel、Text、Markdown）
   - 智能文本分块（按法律条文分割）
   - 增量加载（基于文件哈希缓存）

2. 向量存储：
   - 使用Chroma向量数据库
   - BGE中文Embedding模型

3. 检索与生成：
   - 相似度检索
   - 带上下文的LLM问答
   - 流式响应支持

技术栈：
- LangChain: LLM应用框架
- Chroma: 向量数据库
- HuggingFace Embeddings: 文本向量化
- MarkItDown: 文档格式转换
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from config import settings, get_llm, get_llm_info

logger = logging.getLogger(__name__)

# ==================== 系统提示词 ====================
# 定义LLM的角色和行为要求
PROMPT = ChatPromptTemplate.from_template("""你是一名专业、严谨的AI法律顾问。请仅依据提供的【参考条文】回答用户问题，不得编造不存在的法律条文。
如果参考条文不足以回答问题，请明确说明"参考条文不足"。
【参考条文】
{context}
【用户问题】
{question}
请按照以下结构进行回答：
【法律分析】
- 结合参考条文逐步分析问题
- 说明法律逻辑及适用条件
【适用条文】
- 列出相关条文编号及核心内容
- 如果有多个条文，请逐条说明
【法律结论】
- 给出明确的法律判断
【实务建议】
- 提供可操作的法律建议
- 如涉及风险或注意事项请说明
要求：
1. 回答必须严谨、专业、逻辑清晰
2. 不得虚构法律条文
3. 若条文信息不足，请明确说明
4. 结构化输出
5. 使用简体中文
""")

# 支持的文档扩展名
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


# ==================== 辅助函数 ====================

def _file_md5(path: str) -> str:
    """
    计算文件的MD5哈希值
    
    用于检测文件是否发生变化，实现增量加载。
    
    Args:
        path: 文件路径
    
    Returns:
        文件的MD5哈希值（32位十六进制字符串）
    """
    h = hashlib.md5()
    with open(path, 'rb') as f:
        # 分块读取避免大文件内存溢出
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_cache() -> Dict[str, str]:
    """
    加载文件哈希缓存
    
    从JSON文件读取之前处理过的文件MD5哈希值。
    
    Returns:
        文件路径到MD5哈希的映射字典
    """
    p = Path(settings.FILE_HASH_CACHE)
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def _save_hash_cache(cache: Dict[str, str]) -> None:
    """
    保存文件哈希缓存
    
    将当前处理过的文件MD5哈希值保存到JSON文件。
    
    Args:
        cache: 文件路径到MD5哈希的映射字典
    """
    Path(settings.FILE_HASH_CACHE).write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')


class LawDocumentLoader:
    """
    法律文档加载器
    
    负责从多种格式的法律文档中提取文本，并进行智能分块。
    特性：
    - 支持PDF、Word、Excel、PowerPoint、Text、Markdown格式
    - 智能按法律条文分块
    - 增量加载支持（基于文件MD5哈希）
    - 自动识别法律名称
    """
    
    def __init__(self):
        """初始化文档加载器，配置文本分割器"""
        # 使用递归字符文本分割器，支持多种分隔符
        # 按照法律条文结构优先分割
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n第", "\n##", "\n#", "\n\n", "\n", "。", "；", " ", ""])

    def load_directory(self, directory: str, incremental: bool = True) -> List[Document]:
        """
        加载目录下的所有法律文档
        
        扫描目录下的所有支持格式的文件，进行文本提取和分块。
        
        Args:
            directory: 要扫描的目录路径
            incremental: 是否启用增量加载（默认True，只加载有变化的文件）
        
        Returns:
            提取的文档片段列表
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # 递归获取所有支持格式的文件
        files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        
        # 加载文件哈希缓存，实现增量加载
        hash_cache = _load_hash_cache() if incremental else {}
        new_cache = dict(hash_cache)
        
        docs, skipped = [], 0
        
        for f in files:
            abs_path = str(f.resolve())
            current_md5 = _file_md5(abs_path)
            
            # 如果文件未变化，跳过处理
            if incremental and hash_cache.get(abs_path) == current_md5:
                skipped += 1
                continue
            
            try:
                d = self.load_file(abs_path)
                docs.extend(d)
                new_cache[abs_path] = current_md5
                logger.info(f"✅ {f.name} → {len(d)} 片段")
            except Exception as e:
                logger.error(f"❌ {f.name}: {e}", exc_info=True)
        
        # 保存更新的哈希缓存
        if incremental:
            _save_hash_cache(new_cache)
        
        if skipped:
            logger.info(f"⏭️ 跳过 {skipped} 个未变更文件")
        
        logger.info(f"共加载 {len(docs)} 个新片段")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个法律文档
        
        根据文件扩展名选择合适的解析方式，提取文本并分块。
        
        Args:
            file_path: 文档文件路径
        
        Returns:
            文档片段列表（LangChain Document对象）
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # 根据扩展名选择解析方式
        # 二进制格式（PDF、Word等）使用MarkItDown转换
        # 文本格式直接读取
        text = self._markitdown_convert(file_path) if ext in {'.pdf', '.docx', '.doc', '.pptx', '.xlsx'} else path.read_text(encoding='utf-8', errors='ignore')
        
        if not text.strip():
            return []
        
        # 自动识别法律名称
        law_name = self._get_law_name(path.stem, text)
        # 文本分块
        chunks = self._split_logic(text)
        
        # 转换为LangChain Document对象
        return [Document(page_content=c, metadata={"source": path.name, "law_name": law_name, "article": self._get_article_tag(c) or f"片段{i+1}"}) for i, c in enumerate(chunks) if c.strip()]

    def _markitdown_convert(self, file_path: str) -> str:
        """
        使用MarkItDown将文档转换为纯文本
        
        MarkItDown是一个通用的文档转换库，支持多种格式。
        如果MarkItDown不可用，会回退到备选方案。
        
        Args:
            file_path: 文档路径
        
        Returns:
            提取的纯文本内容
        """
        try:
            return MarkItDown().convert(file_path).text_content
        except ImportError:
            logger.warning("未安装 markitdown，使用备选方案")
        except Exception as e:
            if "MissingDependencyException" not in str(e):
                logger.error(f"MarkItDown 失败: {e}")
        # 回退方案
        return self._fallback(file_path)

    @staticmethod
    def _fallback(file_path: str) -> str:
        """
        备选文档解析方案
        
        当MarkItDown不可用时，使用专用库解析。
        
        Args:
            file_path: 文档路径
        
        Returns:
            提取的文本内容
        """
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.pdf':
                from pypdf import PdfReader
                return "\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)
            if ext in {'.docx', '.doc'}:
                from docx import Document as DocxDoc
                return "\n".join(p.text for p in DocxDoc(file_path).paragraphs)
        except Exception as e:
            logger.warning(f"备选解析失败: {e}")
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')

    def _split_logic(self, text: str) -> List[str]:
        """
        智能文本分块逻辑
        
        优先按法律条文（"第X条"）分割，保持法律条文的完整性。
        如果文本中没有明显的条文结构，则使用通用的递归分割器。
        
        Args:
            text: 待分割的文本
        
        Returns:
            分割后的文本块列表
        """
        # 匹配法律条文的正则表达式
        pat = r'(第[零一二三四五六七八九十百千]+条)'
        parts = re.split(pat, text)
        
        # 如果分割后有多个部分，说明文本包含法律条文
        if len(parts) > 3:
            chunks, cur = [], ""
            for p in parts:
                if re.match(pat, p):
                    # 遇到新条文，保存之前的块
                    if cur.strip() and len(cur) > 20:
                        chunks.append(cur.strip())
                    cur = p
                else:
                    cur += p
                # 如果当前块过大，使用通用分割器继续分割
                if len(cur) > settings.CHUNK_SIZE * 1.5:
                    chunks.extend(self.splitter.split_text(cur))
                    cur = ""
            # 保存最后一个块
            if cur.strip():
                chunks.append(cur.strip())
            return [c for c in chunks if len(c.strip()) > 20]
        
        # 无明显条文结构，使用通用分割
        return self.splitter.split_text(text)

    @staticmethod
    def _get_law_name(stem: str, content: str) -> str:
        """
        从文档内容中提取法律名称
        
        优先从文档第一行提取法律名称，否则使用文件名。
        
        Args:
            stem: 文件名（不含扩展名）
            content: 文档内容
        
        Returns:
            识别的法律名称
        """
        lines = content.strip().split('\n')
        if not lines:
            return stem
        
        first_line = re.sub(r'^[#\s*]+', '', lines[0]).strip()
        # 法律名称通常包含"法"、"条例"、"规定"、"办法"等关键词
        if 5 < len(first_line) < 50 and any(k in first_line for k in ('法', '条例', '规定', '办法')):
            return first_line
        # 清理文件名中的特殊字符
        return re.sub(r'[-_\s\d]+', '', stem) or stem

    @staticmethod
    def _get_article_tag(text: str) -> str:
        """
        从文本中提取条文编号
        
        识别"第X条"格式的条文编号。
        
        Args:
            text: 文本内容
        
        Returns:
            条文编号（如"第123条"），未识别到则返回空字符串
        """
        m = re.search(r'第([零一二三四五六七八九十百千]+)条', text)
        return f"第{m.group(1)}条" if m else ""


class RAGEngine:
    """
    RAG引擎核心类
    
    整合了向量检索和LLM生成的全流程：
    1. 初始化：加载Embedding模型、LLM和向量数据库
    2. 文档管理：添加、删除法律文档
    3. 问答检索：向量检索 + LLM生成回答
    
    核心流程：
    用户问题 → 向量化 → 向量检索 → 找到相关法律条文 → 
    构建Prompt → LLM生成回答 → 返回结果
    """
    
    def __init__(self):
        """初始化RAG引擎的各组件为None"""
        self.vectorstore: Optional[Chroma] = None
        self.llm = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.is_initialized = False
        self.is_loading = False
        self.doc_count = 0
        self.law_names: List[str] = []

    def initialize(self) -> None:
        """
        同步初始化RAG引擎
        """
        logger.info("🚀 初始化 RAG 引擎...")
        
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"📍 使用设备: {device}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': device, 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        try:
            self.llm = get_llm()
        except ValueError as e:
            logger.error(f"❌ LLM 配置错误: {e}")
            raise
        
        db = settings.CHROMA_DB_PATH
        if os.path.exists(db) and os.listdir(db):
            self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings, collection_name="laws")
            self.doc_count = len(self.vectorstore.get()['ids'])
            self._refresh_names()
            logger.info(f"📂 加载向量库: {self.doc_count} 片段")
        else:
            self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings, collection_name="laws")
            self.doc_count = 0
            logger.info("📭 向量库为空，请上传法律文档")
        
        self.is_initialized = True
        logger.info("✅ RAG 引擎就绪")

    async def initialize_async(self) -> None:
        """
        异步初始化RAG引擎
        
        将Embedding模型加载放入线程池，避免阻塞事件循环。
        """
        self.is_loading = True
        logger.info("🚀 异步初始化 RAG 引擎...")
        
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"📍 使用设备: {device}")
        
        def load_embeddings():
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': device, 'local_files_only': False},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )
        
        self.embeddings = await asyncio.to_thread(load_embeddings)
        
        try:
            self.llm = get_llm()
        except ValueError as e:
            logger.error(f"❌ LLM 配置错误: {e}")
            self.is_loading = False
            raise
        
        db = settings.CHROMA_DB_PATH
        if os.path.exists(db) and os.listdir(db):
            self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings, collection_name="laws")
            self.doc_count = len(self.vectorstore.get()['ids'])
            self._refresh_names()
            logger.info(f"📂 加载向量库: {self.doc_count} 片段")
        else:
            self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings, collection_name="laws")
            self.doc_count = 0
            logger.info("📭 向量库为空，请上传法律文档")
        
        self.is_initialized = True
        self.is_loading = False
        logger.info("✅ RAG 引擎就绪")

    def retriever(self):
        """
        创建向量检索器
        
        Returns:
            配置好的Chroma检索器，使用相似度搜索，返回top-k结果
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": settings.TOP_K})

    @staticmethod
    def context(docs: List[Document]) -> str:
        """
        构建检索上下文
        
        将检索到的法律条文格式化为LLM的输入上下文。
        
        Args:
            docs: 检索到的Document列表
        
        Returns:
            格式化后的上下文字符串
        """
        return "\n\n".join(f"【{d.metadata.get('law_name','')} {d.metadata.get('article','')}】\n{d.page_content}" for d in docs)

    async def query(self, question: str) -> Dict[str, Any]:
        if self.vectorstore is None or self.llm is None:
            raise RuntimeError("RAG引擎未初始化")
        
        docs = await asyncio.to_thread(self.retriever().invoke, question)
        
        if not docs:
            return {"answer": "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。", "sources": [], "question": question, "doc_count": 0}
        
        prompt = PROMPT.format_messages(context=self.context(docs), question=question)
        resp = await self.llm.ainvoke(prompt)
        
        return {"answer": resp.content, "sources": self.sources(docs), "question": question, "doc_count": len(docs)}

    @staticmethod
    def sources(docs: List[Document]) -> List[Dict[str, str]]:
        """
        提取文档来源信息
        
        从检索到的文档中提取来源信息，去重后返回。
        
        Args:
            docs: Document列表
        
        Returns:
            来源信息列表，包含law_name、article、content、source_file
        """
        seen, out = set(), []
        for d in docs:
            # 使用法律名称+条文编号作为唯一键去重
            k = f"{d.metadata.get('law_name')}-{d.metadata.get('article')}"
            if k not in seen:
                seen.add(k)
                out.append({
                    "law_name": d.metadata.get("law_name", ""),
                    "article": d.metadata.get("article", ""),
                    # 截取内容摘要（200字符）
                    "content": d.page_content[:200] + ("..." if len(d.page_content) > 200 else ""),
                    "source_file": d.metadata.get("source", "")
                })
        return out

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        同步添加文档到向量数据库
        
        解析文档、分割文本、批量添加到向量数据库。
        
        Args:
            file_path: 文档路径
        
        Returns:
            包含file、chunks_added、total_chunks、law_names的字典
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        
        # 加载并解析文档
        docs = LawDocumentLoader().load_file(file_path)
        if not docs:
            raise ValueError("文档解析失败或内容为空")
        
        # 批量添加到向量数据库（每批32个）
        batch_size = 32
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"📥 处理进度: {min(i + batch_size, len(docs))}/{len(docs)}")
        
        # 更新统计信息
        self.doc_count += len(docs)
        self._refresh_names()
        
        return {"file": Path(file_path).name, "chunks_added": len(docs), "total_chunks": self.doc_count, "law_names": self.law_names}

    def _refresh_names(self) -> None:
        """
        刷新法律名称列表
        
        从向量数据库中获取所有不同的法律名称。
        """
        if self.vectorstore is None:
            self.law_names = []
            return
        try:
            results = self.vectorstore.get(include=["metadatas"])
            # 使用集合去重
            self.law_names = sorted({m["law_name"] for m in results.get("metadatas", []) if m and isinstance(m, dict) and "law_name" in m})
        except Exception as e:
            logger.warning(f"刷新法律名称失败: {e}")
            self.law_names = []

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.is_initialized,
            "loading": self.is_loading,
            "doc_count": self.doc_count,
            "law_names": self.law_names,
            "llm_info": get_llm_info(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "top_k": settings.TOP_K
        }

    async def delete_law(self, law_name: str) -> Dict[str, Any]:
        """
        删除指定法律的所有文档
        
        Args:
            law_name: 要删除的法律名称
        
        Returns:
            删除结果信息
        """
        if not law_name:
            raise ValueError("法律名称不能为空")
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        
        try:
            # 查找该法律的所有文档
            results = self.vectorstore.get(where={"law_name": law_name})
            if not results or not results.get("ids"):
                return {"success": False, "message": f"未找到法律: {law_name}", "deleted_count": 0}
            
            # 删除这些文档
            self.vectorstore.delete(ids=results["ids"])
            self.doc_count -= len(results["ids"])
            self._refresh_names()
            
            return {
                "success": True,
                "message": f"成功删除法律: {law_name}",
                "deleted_count": len(results["ids"]),
                "remaining_docs": self.doc_count,
                "law_names": self.law_names
            }
        except Exception as e:
            logger.error(f"删除失败: {e}")
            raise ValueError(f"删除失败: {str(e)}")


# 创建全局RAG引擎单例实例
rag_engine = RAGEngine()