"""
AI 法律顾问 — FastAPI 后端服务

本文件实现了一个基于RAG（检索增强生成）技术的AI法律顾问系统后端。
主要功能包括：
- 法律文档上传与管理
- 基于向量检索的法律问答
- 流式响应支持

依赖技术：
- FastAPI: Web框架
- Chroma: 向量数据库
- LangChain: LLM应用开发框架
"""

# 导入必要的标准库和第三方库
import asyncio
import json
import logging
import os
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from rag_engine import history_manager, get_session_history
from langchain_core.messages import messages_to_dict

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
    initialized: bool
    doc_count: int
    law_names: List[str]
    llm_info: Dict[str, Any]
    embedding_model: str
    chunk_size: int
    top_k: int


# 允许上传的文件扩展名集合
# 仅支持PDF和Word格式的法律文档
ALLOWED_EXT = {'.pdf', '.docx'}

# 线程池执行器，用于处理同步的文档处理任务
# 最多同时处理2个文档，避免阻塞异步事件循环
executor = ThreadPoolExecutor(max_workers=2)

# 上传并发信号量：同一时刻最多允许 2 个文件在处理
# 超出的请求会等待，而不是同时把多个大文件塞进内存
upload_semaphore = asyncio.Semaphore(2)

# 初始化完成事件，用于通知系统RAG引擎已就绪
init_event = asyncio.Event()


@asynccontextmanager
async def lifespan(_: FastAPI):
    async def init_rag():
        try:
            from rag_engine import rag_engine
            await rag_engine.initialize_async()
            init_event.set()
        except Exception as e:
            logger.error(f"启动失败: {e}", exc_info=True)
            init_event.set()
    
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
        result = await rag_engine.query(req.question, req.session_id)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    问答API（流式响应）
    
    与query API类似，但使用Server-Sent Events (SSE)进行流式输出，
    实现打字机效果的实时回答。
    """
    from rag_engine import rag_engine
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")

    async def gen():
        async for chunk in rag_engine.astream_query(req.question, req.session_id):
            if isinstance(chunk, dict):
                if "error" in chunk:
                    yield f"data: {chunk['error']}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield f"data: [METADATA]{json.dumps(chunk)}\n\n"
            else:
                yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    文档上传API

    接受用户上传的法律文档（PDF、docx格式），
    自动解析、分割并存储到向量数据库中。

    内存优化策略：
    - 分块流式写入临时文件，单次最多读取 1 MB，避免整个文件驻留内存
    - 大小校验在写入过程中累计，超限立即中止并删除临时文件
    - upload_semaphore 限制同时处理的文件数，防止并发上传撑爆内存

    请求:
        file: 上传的文件，支持 .pdf, .docx

    响应:
        success: 是否成功
        chunks_added: 新增的文本片段数
        total_chunks: 数据库中总片段数
        law_names: 涉及的法律名称
    """
    from rag_engine import rag_engine
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式: {ext}")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)

    async with upload_semaphore:
        try:
            # 分块流式写入，每次最多读取 1 MB，不把整个文件载入内存
            CHUNK = 1024 * 1024  # 1 MB
            total = 0
            with open(tmp_path, 'wb') as f:
                while True:
                    chunk = await file.read(CHUNK)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > settings.MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            413,
                            f"文件过大，最大允许 {settings.MAX_UPLOAD_SIZE // 1024 // 1024} MB"
                        )
                    f.write(chunk)

            # 文档解析在线程池中执行，避免阻塞事件循环
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, rag_engine.add_document, tmp_path)
            return UploadResponse(success=True, message=f"'{filename}' 加载成功", **result)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"文档处理失败: {e}")
        finally:
            if os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except OSError as e:
                    logger.warning(f"清理临时目录失败 {tmp_dir}: {e}")


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
async def get_history(limit: int = 100, offset: int = 0):
    """获取会话列表"""
    sessions = history_manager.list_sessions(limit=limit, offset=offset)
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/api/history/{session_id}")
async def get_session_messages(session_id: str):
    """获取指定会话的消息历史"""
    chat_history = get_session_history(session_id)
    messages = messages_to_dict(chat_history.messages)
    if not messages:
        raise HTTPException(404, "会话不存在")
    return {"session_id": session_id, "messages": messages}


@app.delete("/api/history/{session_id}")
async def delete_session_history(session_id: str):
    """删除指定会话历史"""
    if not history_manager.delete_session(session_id):
        raise HTTPException(404, "会话不存在")
    return {"success": True, "message": f"会话 {session_id} 已删除"}


@app.delete("/api/history")
async def clear_history():
    """清空所有会话历史"""
    count = history_manager.clear_all()
    return {"success": True, "deleted_count": count}


class RenameRequest(BaseModel):
    """重命名请求模型"""
    title: str = Field(..., min_length=1, max_length=50)


@app.patch("/api/history/{session_id}")
async def rename_session(session_id: str, req: RenameRequest):
    """重命名会话"""
    if not history_manager.rename_session(session_id, req.title):
        raise HTTPException(404, "会话不存在")
    return {"success": True, "session_id": session_id, "title": req.title}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)