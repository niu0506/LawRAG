"""
AI 法律顾问 — FastAPI 后端服务

本文件实现了一个基于RAG（检索增强生成）技术的AI法律顾问系统后端。
主要功能包括：
- 法律文档上传与管理
- 基于向量检索的法律问答
- 对话历史记录存储
- 流式响应支持

依赖技术：
- FastAPI: Web框架
- Chroma: 向量数据库
- LangChain: LLM应用开发框架
- SQLite: 历史记录存储
"""

# 导入必要的标准库和第三方库
import asyncio
import json
import logging
import os
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


"""
数据模型定义

使用Pydantic定义请求/响应的数据结构，用于API参数验证和自动文档生成。
"""

# Pydantic模型：法律来源文档
# 用于返回引用的法律条文信息
class SourceDocument(BaseModel):
    """法律来源文档模型"""
    law_name: str          # 法律名称（如《民法典》、《刑法》等）
    article: str           # 条文编号（如"第123条"）
    content: str           # 条文内容摘要
    source_file: str       # 来源文件名


# Pydantic模型：查询请求
class QueryRequest(BaseModel):
    """用户问题请求模型"""
    question: str = Field(..., min_length=2, max_length=1000)  # 用户问题，长度限制2-1000字符
    session_id: Optional[str] = None  # 会话ID，用于关联历史记录


# Pydantic模型：查询响应
class QueryResponse(BaseModel):
    """问答响应模型"""
    answer: str                      # AI生成的回答
    sources: List[SourceDocument]    # 引用的法律条文列表
    question: str                    # 用户问题（回显）
    doc_count: int = 0               # 检索到的文档数量
    session_id: str = ""             # 会话ID
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())  # 响应时间戳


# Pydantic模型：上传响应
class UploadResponse(BaseModel):
    """文档上传响应模型"""
    success: bool            # 上传是否成功
    file: str               # 上传的文件名
    chunks_added: int       # 新增的文本片段数量
    total_chunks: int       # 总文本片段数量
    law_names: List[str]    # 涉及的法律名称列表
    message: str            # 响应消息


# Pydantic模型：系统状态响应
class StatusResponse(BaseModel):
    """系统状态响应模型"""
    initialized: bool                       # RAG引擎是否已初始化
    doc_count: int                          # 当前存储的文档片段数量
    law_names: List[str]                    # 已加载的法律名称列表
    llm_info: Dict[str, Any]                # LLM提供商信息
    embedding_model: str                    # 使用的embedding模型名称
    chunk_size: int                         # 文本分块大小
    top_k: int                              # 检索时返回的top-k结果数


# Pydantic模型：历史记录
class HistoryRecord(BaseModel):
    """历史问答记录模型"""
    id: int                          # 记录ID
    session_id: str                  # 会话ID
    question: str                    # 用户问题
    answer: str                      # AI回答
    sources: List[SourceDocument]    # 引用的法律条文
    timestamp: str                   # 记录时间戳


# 允许上传的文件扩展名列表
# 支持PDF、Word文档、PowerPoint、Excel、文本文件和Markdown
ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


class HistoryDB:
    """
    历史记录数据库管理类
    
    使用SQLite存储用户的问答历史记录，支持以下功能：
    - 保存问答记录
    - 查询历史记录（按会话或全部）
    - 删除特定会话的历史记录
    - 清空所有历史记录
    """
    
    def __init__(self):
        """初始化数据库连接并创建必要的表结构"""
        self._init_db()

    def _init_db(self):
        """创建历史记录表和索引"""
        with self._conn() as conn:
            # 创建history表，包含会话ID、问题、回答、来源和时间戳
            conn.execute("""CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, question TEXT NOT NULL,
                answer TEXT NOT NULL, sources TEXT DEFAULT '[]', timestamp TEXT NOT NULL)""")
            # 创建会话ID索引，加速按会话查询
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")
            # 创建时间戳索引，加速排序查询
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON history(timestamp)")

    @contextmanager
    def _conn(self):
        """数据库连接上下文管理器"""
        # 连接到SQLite数据库
        conn = sqlite3.connect(settings.HISTORY_DB_PATH)
        try:
            yield conn
            conn.commit()  # 提交事务
        finally:
            conn.close()   # 关闭连接

    def save(self, session_id: str, question: str, answer: str, sources: List[SourceDocument]) -> int:
        """
        保存问答记录到数据库
        
        Args:
            session_id: 会话ID
            question: 用户问题
            answer: AI回答
            sources: 引用的法律条文列表
        
        Returns:
            新记录的ID
        """
        with self._conn() as conn:
            cur = conn.cursor()
            # 插入新记录，来源以JSON格式存储
            cur.execute("INSERT INTO history (session_id, question, answer, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
                (session_id, question, answer, json.dumps([s.model_dump() for s in sources]), datetime.now().isoformat()))
            return cur.lastrowid or 0

    def get_history(self, session_id: Optional[str] = None, limit: int = 100, offset: int = 0) -> tuple:
        """
        获取历史记录
        
        Args:
            session_id: 可选的会话ID筛选条件
            limit: 返回记录数量限制
            offset: 偏移量，用于分页
        
        Returns:
            (记录列表, 总记录数)的元组
        """
        with self._conn() as conn:
            cur = conn.cursor()
            if session_id:
                # 按会话ID查询
                cur.execute("SELECT COUNT(*) FROM history WHERE session_id = ?", (session_id,))
                total = cur.fetchone()[0]
                cur.execute("SELECT id, session_id, question, answer, sources, timestamp FROM history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?", (session_id, limit, offset))
            else:
                # 查询所有记录
                cur.execute("SELECT COUNT(*) FROM history")
                total = cur.fetchone()[0]
                cur.execute("SELECT id, session_id, question, answer, sources, timestamp FROM history ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset))
            # 将查询结果转换为HistoryRecord对象列表
            records = [HistoryRecord(id=r[0], session_id=r[1], question=r[2], answer=r[3],
                        sources=[SourceDocument(**s) for s in json.loads(r[4])], timestamp=r[5]) for r in cur.fetchall()]
            return records, total

    def delete_session(self, session_id: str) -> int:
        """删除指定会话的所有历史记录"""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
            return cur.rowcount

    def clear_all(self) -> int:
        """清空所有历史记录并返回删除的记录数"""
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM history")
            count = cur.fetchone()[0]
            cur.execute("DELETE FROM history")
            return count


# 创建全局单例实例
history_db = HistoryDB()
# 线程池执行器，用于后台处理文件上传等耗时任务
executor = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def init_rag():
        try:
            from rag_engine import rag_engine
            await rag_engine.initialize_async()
        except Exception as e:
            logger.error(f"启动失败: {e}", exc_info=True)
    
    asyncio.create_task(init_rag())
    yield


# 创建FastAPI应用实例
app = FastAPI(title="AI 法律顾问", version="1.0.0", lifespan=lifespan)

# 添加CORS中间件，允许跨域请求
# settings.CORS_ORIGINS控制允许的来源，默认为["*"]允许所有
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 静态文件服务配置
# 将当前目录下的文件作为静态资源提供，用于前端页面
frontend_path = Path(__file__).parent
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """根路由，返回前端HTML页面"""
    f = frontend_path / "index.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"status": "running"})


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """
    获取系统状态API
    
    返回当前RAG引擎的初始化状态、文档数量、使用的模型等信息。
    用于前端检查系统是否就绪。
    """
    from rag_engine import rag_engine
    return rag_engine.get_status()


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    问答API（非流式）
    
    接收用户问题，调用RAG引擎进行向量检索和LLM生成，
    返回完整的回答和引用的法律条文。
    
    请求体:
        question: 用户问题
        session_id: 可选的会话ID
    
    响应:
        answer: AI生成的回答
        sources: 引用条文列表
        doc_count: 检索到的文档数
    """
    from rag_engine import rag_engine
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")
    try:
        result = await rag_engine.query(req.question)
        # 生成或使用传入的会话ID
        session_id = req.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # 将源文档转换为模型对象
        sources = [SourceDocument(**s) for s in result.get("sources", [])]
        # 保存到历史记录
        history_db.save(session_id=session_id, question=req.question, answer=result["answer"], sources=sources)
        result["session_id"] = session_id
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    问答API（流式响应）
    
    与query API类似，但使用Server-Sent Events (SSE)进行流式输出，
    实现打字机效果的实时回答。
    
    请求体:
        question: 用户问题
        session_id: 可选的会话ID
    
    响应:
        流式的文本块，以data:开头的SSE格式
        最后发送 [METADATA] 包含sources等信息
    """
    from rag_engine import rag_engine, PROMPT
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")

    async def gen():
        if rag_engine.vectorstore is None or rag_engine.llm is None:
            yield "data: [ERROR]RAG引擎未初始化\n\n"
            return
        
        docs = await asyncio.to_thread(rag_engine._retriever().invoke, req.question)
        if not docs:
            yield "data: 未找到相关法律条文。\n\n"
            yield "data: [DONE]\n\n"
            return
        
        sources = rag_engine._sources(docs)
        context = rag_engine._context(docs)
        full_answer = ""
        
        # 流式生成
        prompt = PROMPT.format_messages(context=context, question=req.question)
        async for chunk in rag_engine.llm.astream(prompt):
            if chunk.content:
                full_answer += chunk.content
                yield f"data: {chunk.content}\n\n"
        
        # 保存历史记录
        session_id = req.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        source_docs = [SourceDocument(**s) for s in sources]
        history_db.save(session_id=session_id, question=req.question, answer=full_answer, sources=source_docs)
        
        metadata = json.dumps({"session_id": session_id, "sources": sources, "doc_count": len(docs)})
        yield f"data: [METADATA]{metadata}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    文档上传API
    
    接受用户上传的法律文档（PDF、Word、Text等格式），
    自动解析、分割并存储到向量数据库中。
    
    请求:
        file: 上传的文件，支持.pdf, .docx, .doc, .pptx, .xlsx, .txt, .md
    
    响应:
        success: 是否成功
        chunks_added: 新增的文本片段数
        total_chunks: 数据库中总片段数
        law_names: 涉及的法律名称
    """
    from rag_engine import rag_engine
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")

    # 读取文件内容
    content = await file.read()
    # 检查文件大小是否超过限制
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"文件过大，最大允许 {settings.MAX_UPLOAD_SIZE // 1024 // 1024}MB")

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    # 检查文件扩展名是否允许
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式: {ext}")

    # 创建临时目录和临时文件来处理上传的文档
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        # 将内容写入临时文件
        with open(tmp_path, 'wb') as f:
            f.write(content)
        # 使用线程池执行同步的文档处理任务，避免阻塞异步事件循环
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, rag_engine.add_document, tmp_path)
        return UploadResponse(success=True, message=f"'{filename}' 加载成功", **result)
    except Exception as e:
        raise HTTPException(500, f"文档处理失败: {e}")
    finally:
        # 清理临时文件和目录
        for p in [tmp_path, tmp_dir]:
            if os.path.exists(p):
                try:
                    os.unlink(p) if os.path.isfile(p) else os.rmdir(p)
                except OSError as e:
                    logger.warning(f"清理临时文件失败 {p}: {e}")


@app.get("/api/laws")
async def laws():
    """
    获取已加载的法律文档列表
    
    返回当前向量数据库中存储的所有法律文档信息。
    """
    from rag_engine import rag_engine
    return {"laws": rag_engine.law_names, "total": len(rag_engine.law_names), "doc_count": rag_engine.doc_count}


@app.delete("/api/laws/{law_name}")
async def delete_law(law_name: str):
    """
    删除指定法律文档
    
    从向量数据库中删除指定法律名称的所有文档片段。
    
    路径参数:
        law_name: 要删除的法律名称（URL编码）
    """
    from rag_engine import rag_engine
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")
    try:
        return await rag_engine.delete_law(law_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"删除失败: {str(e)}")


@app.delete("/api/rebuild")
async def rebuild(bg: BackgroundTasks):
    """
    重建向量数据库API
    
    删除现有的Chroma向量数据库并在后台重新初始化。
    这在向量数据库损坏或需要重置时使用。
    
    注意: 这是一个异步操作，通过BackgroundTasks在后台执行。
    """
    def _do():
        import shutil
        # 删除现有的向量数据库目录
        if os.path.exists(settings.CHROMA_DB_PATH):
            shutil.rmtree(settings.CHROMA_DB_PATH)
        # 重置RAG引擎状态并重新初始化
        from rag_engine import rag_engine
        rag_engine.is_initialized = False
        rag_engine.initialize()
    # 添加后台任务
    bg.add_task(_do)
    return {"message": "后台重建中"}


@app.get("/api/health")
async def health():
    """
    健康检查API
    
    简单的健康检查端点，返回系统状态和RAG引擎初始化状态。
    用于负载均衡器的健康探测。
    """
    from rag_engine import rag_engine
    return {"status": "ok", "initialized": rag_engine.is_initialized}


@app.get("/api/history")
async def get_history(session_id: Optional[str] = None, limit: int = 100, page: int = 1):
    """
    获取问答历史记录API
    
    查询存储的问答历史，支持分页和会话筛选。
    
    查询参数:
        session_id: 可选的会话ID筛选
        limit: 每页记录数，默认100
        page: 页码，默认1
    """
    offset = (page - 1) * limit
    records, total = history_db.get_history(session_id=session_id, limit=limit, offset=offset)
    return {"records": [r.model_dump() for r in records], "total": total, "page": page, "page_size": limit}


@app.delete("/api/history/{session_id}")
async def delete_session_history(session_id: str):
    """
    删除指定会话的历史记录
    
    路径参数:
        session_id: 要删除的会话ID
    """
    return {"success": True, "deleted": history_db.delete_session(session_id), "session_id": session_id}


@app.delete("/api/history")
async def clear_history():
    """
    清空所有历史记录API
    
    删除所有问答历史记录。此操作不可恢复。
    """
    return {"success": True, "deleted": history_db.clear_all()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)