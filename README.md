# LawRAG - AI 法律顾问

基于 RAG（检索增强生成）技术的智能法律问答系统。

## 功能特性

- **智能法律问答**：基于上传的法律文档进行精准问答
- **多格式文档支持**：PDF、Word、PowerPoint、Excel、Text、Markdown
- **向量检索**：使用 Chroma 向量数据库 + BGE 中文 Embedding 模型
- **流式响应**：支持打字机效果的实时回答
- **历史记录**：自动保存对话历史，支持回溯
- **增量加载**：基于文件哈希的增量文档更新

## 技术栈

- **后端框架**：FastAPI
- **向量数据库**：Chroma
- **Embedding 模型**：BAAI/bge-large-zh-v1.5
- **LLM 框架**：LangChain
- **文档解析**：MarkItDown、pypdf、python-docx

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/niu0506/LawRAG.git
cd LawRAG
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制配置模板并填写你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# LLM 配置（支持 OpenAI 兼容格式的 API）
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4

# HuggingFace Token（用于下载 Embedding 模型）
HF_TOKEN=your_hf_token_here
```

### 4. 下载 Embedding 模型

首次运行需要下载 BGE 中文 Embedding 模型：

```bash
# 设置 HF_TOKEN 后会自动下载
# 或手动下载到本地缓存
```

### 5. 启动服务

```bash
python main.py
```

服务启动后访问 http://localhost:8000

## API 接口

### 问答接口

```bash
# 普通问答
POST /api/query
{
  "question": "合同违约如何处理？",
  "session_id": "optional_session_id"
}

# 流式问答（SSE）
POST /api/query/stream
```

### 文档管理

```bash
# 上传文档
POST /api/upload

# 获取已加载的法律列表
GET /api/laws

# 删除指定法律
DELETE /api/laws/{law_name}
```

### 系统状态

```bash
# 获取系统状态
GET /api/status

# 健康检查
GET /api/health
```

### 历史记录

```bash
# 获取历史记录
GET /api/history?session_id=xxx&limit=100&page=1

# 删除会话历史
DELETE /api/history/{session_id}

# 清空所有历史
DELETE /api/history
```

## 项目结构

```
LawRAG/
├── main.py          # FastAPI 后端服务
├── rag_engine.py    # RAG 引擎核心
├── config.py        # 配置管理
├── index.html       # 前端页面
├── requirements.txt # 依赖列表
├── .env.example     # 环境变量模板
└── db/              # 数据存储目录
    ├── chroma/      # 向量数据库
    └── history.db   # 历史记录
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| LLM_API_KEY | LLM API 密钥 | - |
| LLM_BASE_URL | LLM API 地址 | - |
| LLM_MODEL | 模型名称 | - |
| HF_TOKEN | HuggingFace Token | - |
| EMBEDDING_MODEL | Embedding 模型 | BAAI/bge-large-zh-v1.5 |
| CHUNK_SIZE | 文本分块大小 | 500 |
| CHUNK_OVERLAP | 分块重叠字符数 | 50 |
| TOP_K | 检索返回数量 | 5 |
| HOST | 服务监听地址 | 0.0.0.0 |
| PORT | 服务端口 | 8000 |

## License

MIT