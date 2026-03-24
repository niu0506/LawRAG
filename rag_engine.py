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
import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings, get_llm, get_llm_info

logger = logging.getLogger(__name__)


# ==================== 数据库辅助函数 ====================

_db_lock = threading.Lock()

def _get_db_conn(db_path: str) -> sqlite3.Connection:
    """获取数据库连接的共享函数"""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db_schema(conn: sqlite3.Connection) -> None:
    """初始化数据库表结构"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    """)


# ==================== 对话历史管理 ====================

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """基于 SQLite 的对话历史存储"""
    
    def __init__(self, session_id: str, db_path: str = "./db/history.db"):
        self.session_id = session_id
        self.db_path = db_path
        self.max_history = settings.HISTORY_TURNS
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                _init_db_schema(conn)
    
    @property
    def messages(self) -> List[BaseMessage]:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (self.session_id,))
                rows = conn.execute(
                    "SELECT message FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (self.session_id, self.max_history * 2)
                ).fetchall()
                msgs = messages_from_dict([json.loads(row["message"]) for row in reversed(rows)])
                return msgs
    
    def add_message(self, message: BaseMessage) -> None:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                conn.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (self.session_id,))
                conn.execute(
                    "INSERT INTO messages (session_id, message) VALUES (?, ?)",
                    (self.session_id, json.dumps(messages_to_dict([message])[0], ensure_ascii=False))
                )
                conn.execute(
                    "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (self.session_id,)
                )
    
    def clear(self) -> None:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (self.session_id,))


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, db_path: str = "./db/history.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                _init_db_schema(conn)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC)")
                try:
                    conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
                except sqlite3.OperationalError:
                    pass
    
    def list_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                rows = conn.execute(
                    """SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) as message_count,
                              (SELECT message FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) as last_message
                       FROM sessions s LEFT JOIN messages m ON s.id = m.session_id
                       GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ? OFFSET ?""",
                    (limit, offset)
                ).fetchall()
                results = []
                for row in rows:
                    r = dict(row)
                    last_msg = r.get("last_message")
                    if last_msg:
                        try:
                            msg_data = json.loads(last_msg)
                            content = msg_data.get("content", "")
                            r["last_message"] = content[:50] + ("..." if len(content) > 50 else "")
                        except:
                            r["last_message"] = ""
                    results.append(r)
                return results
    
    def delete_session(self, session_id: str) -> bool:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                return cursor.rowcount > 0
    
    def rename_session(self, session_id: str, title: str) -> bool:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                cursor = conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id)
                )
                return cursor.rowcount > 0
    
    def clear_all(self) -> int:
        with _db_lock:
            with _get_db_conn(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM sessions")
                return cursor.rowcount


def get_session_history(session_id: str) -> SQLiteChatMessageHistory:
    """获取会话历史的工厂函数"""
    return SQLiteChatMessageHistory(session_id)


history_manager = HistoryManager()

# ==================== 系统提示词 ====================
# 定义LLM的角色和行为要求
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业、严谨的AI法律顾问。请仅依据提供的【参考条文】回答用户问题，不得编造不存在的法律条文。
如果参考条文不足以回答问题，请明确说明"参考条文不足"。
【参考条文】
{context}
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
5. 使用简体中文"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 支持的文档扩展名
SUPPORTED_EXTENSIONS = {'.pdf', '.docx'}


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
        
        text = self._load_document(file_path, ext)
        if not text.strip():
            return []
        
        law_name = self._get_law_name(path.stem, text)
        chunks = self._split_logic(text)
        
        return [Document(page_content=c, metadata={"source": path.name, "law_name": law_name, "article": self._get_article_tag(c) or f"片段{i+1}"}) for i, c in enumerate(chunks) if c.strip()]

    def _load_document(self, file_path: str, ext: str) -> str:
        """
        根据文件类型加载文档内容
        
        Args:
            file_path: 文档路径
            ext: 文件扩展名（小写）
        
        Returns:
            提取的纯文本内容
        """
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                return '\n'.join(doc.page_content for doc in docs)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                return '\n'.join(doc.page_content for doc in docs)
            else:
                return Path(file_path).read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"文档解析失败: {e}")
            return ""

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
        self.chain_with_history = None
        self.is_initialized = False
        self.is_loading = False
        self.doc_count = 0
        self.law_names: List[str] = []

    @staticmethod
    def _get_device() -> str:
        """获取计算设备"""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _create_embeddings(device: str) -> HuggingFaceEmbeddings:
        """创建Embedding模型"""
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': device, 'local_files_only': False},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )

    def _setup_vectorstore(self) -> None:
        """设置向量数据库"""
        db = settings.CHROMA_DB_PATH
        if os.path.exists(db) and os.listdir(db):
            self.vectorstore = Chroma(
                persist_directory=db,
                embedding_function=self.embeddings,
                collection_name="laws"
            )
            self.doc_count = len(self.vectorstore.get()['ids'])
            self._refresh_names()
            logger.info(f"📂 加载向量库: {self.doc_count} 片段")
        else:
            self.vectorstore = Chroma(
                persist_directory=db,
                embedding_function=self.embeddings,
                collection_name="laws"
            )
            self.doc_count = 0
            logger.info("📭 向量库为空，请上传法律文档")

    def initialize(self) -> None:
        """同步初始化RAG引擎"""
        logger.info("🚀 初始化 RAG 引擎...")
        
        device = self._get_device()
        logger.info(f"📍 使用设备: {device}")
        
        self.embeddings = self._create_embeddings(device)
        
        try:
            self.llm = get_llm()
        except ValueError as e:
            logger.error(f"❌ LLM 配置错误: {e}")
            raise
        
        self._setup_vectorstore()
        self.is_initialized = True
        self._build_chain()
        logger.info("✅ RAG 引擎就绪")

    def _build_chain(self) -> None:
        """构建带历史记录的 chain"""
        chain = PROMPT | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history"
        )

    async def initialize_async(self) -> None:
        """异步初始化RAG引擎，将Embedding模型加载放入线程池"""
        self.is_loading = True
        logger.info("🚀 异步初始化 RAG 引擎...")
        
        device = self._get_device()
        logger.info(f"📍 使用设备: {device}")
        
        self.embeddings = await asyncio.to_thread(self._create_embeddings, device)
        
        try:
            self.llm = get_llm()
        except ValueError as e:
            logger.error(f"❌ LLM 配置错误: {e}")
            self.is_loading = False
            raise
        
        await asyncio.to_thread(self._setup_vectorstore)
        self.is_initialized = True
        self.is_loading = False
        self._build_chain()
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

    async def query(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        异步执行问答检索
        
        完整流程：
        1. 接收用户问题
        2. 调用异步检索获取相关法律条文
        3. 构建上下文并调用带历史记录的LLM chain
        4. 返回回答、来源文档和会话ID
        
        Args:
            question: 用户提出的法律问题
            session_id: 会话ID，用于关联对话历史
        
        Returns:
            包含以下键的字典：
            - answer: LLM生成的回答
            - sources: 引用来源列表
            - question: 原始问题
            - doc_count: 检索到的文档数
            - session_id: 会话ID
        """
        if self.vectorstore is None or self.chain_with_history is None:
            raise RuntimeError("RAG引擎未初始化")
        
        docs = await self.aretrieve(question)
        
        if not docs:
            return {"answer": "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。", "sources": [], "question": question, "doc_count": 0}
        
        if not session_id:
            session_id = f"session_{os.urandom(4).hex()}"
        
        context = self.context(docs)
        resp = await self.chain_with_history.ainvoke(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        )
        
        return {"answer": resp.content, "sources": self.sources(docs), "question": question, "doc_count": len(docs), "session_id": session_id}

    async def aretrieve(self, question: str) -> List[Document]:
        """
        异步向量检索
        
        将用户问题转换为向量，在向量数据库中检索相似度最高的法律条文。
        
        Args:
            question: 用户问题
        
        Returns:
            检索到的相关文档列表（Document对象列表）
        
        Raises:
            RuntimeError: 向量存储未初始化时抛出
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        return await asyncio.to_thread(self.retriever().invoke, question)

    async def astream_query(self, question: str, session_id: Optional[str] = None):
        """
        流式问答生成器
        
        Yields:
            str: 流式输出的文本块，或 {"error": "..."} 错误信息
        """
        if not self.is_initialized or self.chain_with_history is None:
            yield {"error": "RAG引擎未初始化"}
            return
        
        docs = await self.aretrieve(question)
        if not docs:
            yield {"error": "未找到相关法律条文"}
            return
        
        if not session_id:
            session_id = f"session_{os.urandom(4).hex()}"
        
        context = self.context(docs)
        sources = self.sources(docs)
        
        async for chunk in self.chain_with_history.astream(
            {"question": question, "context": context},
            config={"configurable": {"session_id": session_id}}
        ):
            if chunk.content:
                yield chunk.content
        
        yield {"session_id": session_id, "sources": sources, "doc_count": len(docs)}

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
        如果法律已存在则跳过，避免重复向量化。
        
        Args:
            file_path: 文档路径
        
        Returns:
            包含file、chunks_added、total_chunks、law_names、skipped的字典
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        
        # 计算文件哈希，用于去重
        file_hash = _file_md5(file_path)
        
        # 检查向量库中是否已有相同哈希的文档
        existing = self.vectorstore.get(where={"file_hash": file_hash})
        if existing and existing.get("ids"):
            logger.info(f"⏭️ 文件已存在，跳过: {Path(file_path).name} (hash: {file_hash[:8]}...)")
            return {
                "file": Path(file_path).name,
                "chunks_added": 0,
                "total_chunks": self.doc_count,
                "law_names": self.law_names,
                "skipped": True,
                "reason": f"文件内容已存在 (hash: {file_hash[:8]}...)"
            }
        
        # 加载并解析文档
        docs = LawDocumentLoader().load_file(file_path)
        if not docs:
            raise ValueError("文档解析失败或内容为空")
        
        # 为每个文档添加文件哈希元数据
        for doc in docs:
            doc.metadata["file_hash"] = file_hash
        
        # 批量添加到向量数据库（每批32个）
        batch_size = 32
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.vectorstore.add_documents(batch)
            logger.info(f"📥 处理进度: {min(i + batch_size, len(docs))}/{len(docs)}")
        
        # 更新统计信息
        self.doc_count += len(docs)
        self._refresh_names()
        
        return {"file": Path(file_path).name, "chunks_added": len(docs), "total_chunks": self.doc_count, "law_names": self.law_names, "skipped": False}

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
        """
        获取RAG引擎状态信息
        
        返回当前引擎的完整状态，用于前端显示和健康检查。
        
        Returns:
            包含以下信息的字典：
            - initialized: 是否已初始化
            - loading: 是否正在加载
            - doc_count: 文档片段总数
            - law_names: 法律名称列表
            - llm_info: LLM模型信息
            - embedding_model: Embedding模型名称
            - chunk_size: 文本块大小
            - top_k: 检索返回结果数
        """
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