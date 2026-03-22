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
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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
    initialized: bool
    doc_count: int
    law_names: List[str]
    llm_info: Dict[str, Any]
    embedding_model: str
    chunk_size: int
    top_k: int


ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


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
        sources = [SourceDocument(**s) for s in result.get("sources", [])]
        result["session_id"] = req.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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
        
        docs = await asyncio.to_thread(rag_engine.retriever().invoke, req.question)
        if not docs:
            yield "data: 未找到相关法律条文。\n\n"
            yield "data: [DONE]\n\n"
            return
        
        sources = rag_engine.sources(docs)
        context = rag_engine.context(docs)
        full_answer = ""
        
        # 流式生成
        prompt = PROMPT.format_messages(context=context, question=req.question)
        async for chunk in rag_engine.llm.astream(prompt):
            if chunk.content:
                full_answer += chunk.content
                yield f"data: {chunk.content}\n\n"
        
        session_id = req.session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)