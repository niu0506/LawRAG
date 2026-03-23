# LawRAG

基于 RAG 技术的智能法律问答系统。

## 功能

- 智能法律问答 - 基于上传文档精准回答
- 多轮对话 - 支持上下文连续问答
- 对话历史 - SQLite 持久化存储
- 多格式支持 - PDF、Word 文档
- 向量检索 - Chroma + BGE 中文 Embedding
- 流式响应 - SSE 实时输出
- 增量加载 - 文件哈希去重

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 向量数据库 | Chroma |
| Embedding | BAAI/bge-large-zh-v1.5 |
| LLM框架 | LangChain |

## 快速开始

```bash
# 克隆项目
git clone https://github.com/niu0506/LawRAG.git
cd LawRAG

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填写 API 密钥

# 启动服务
python main.py
```

访问 http://localhost:8000

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `LLM_API_KEY` | LLM API 密钥 | ✓ |
| `LLM_BASE_URL` | LLM API 地址 | ✓ |
| `LLM_MODEL` | 模型名称 | ✓ |
| `HF_TOKEN` | HuggingFace Token | |
| `EMBEDDING_MODEL` | Embedding 模型 | 默认: BAAI/bge-large-zh-v1.5 |
| `CHUNK_SIZE` | 文本分块大小 | 默认: 500 |
| `CHUNK_OVERLAP` | 分块重叠 | 默认: 50 |
| `TOP_K` | 检索数量 | 默认: 5 |
| `HISTORY_TURNS` | 多轮对话轮数 | 默认: 5 |

## API

### 问答

```
POST /api/query          # 普通问答
POST /api/query/stream   # 流式问答 (SSE)
```

### 文档管理

```
POST   /api/upload       # 上传文档
GET    /api/laws         # 法律列表
DELETE /api/laws/{name}  # 删除法律
```

### 系统

```
GET  /api/status    # 系统状态
GET  /api/health    # 健康检查
POST /api/rebuild   # 重建向量数据库
```

### 历史记录

```
GET    /api/history           # 会话列表
GET    /api/history/{id}      # 会话消息
DELETE /api/history/{id}      # 删除会话
DELETE /api/history           # 清空历史
```

## 项目结构

```
LawRAG/
├── main.py          # FastAPI 服务
├── rag_engine.py    # RAG 引擎 + 对话历史
├── config.py        # 配置管理
├── index.html       # 前端页面
├── requirements.txt
└── db/
    ├── chroma/      # 向量数据库
    └── history.db   # 对话历史
```

## License

MIT