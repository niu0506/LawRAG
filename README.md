# LawRAG

基于 RAG 技术的智能法律问答系统。

## 功能特性

- **智能法律问答** - 基于上传的法律文档精准回答问题
- **多轮对话** - 支持上下文连续问答，自动保持对话历史
- **对话历史管理** - SQLite 持久化存储，支持查看和删除历史会话
- **多格式文档支持** - 支持 PDF、Word 文档上传
- **向量检索** - 使用 Chroma 向量数据库 + BGE 中文 Embedding 模型
- **流式响应** - SSE 实时输出，打字机效果
- **增量加载** - 基于文件哈希去重，避免重复处理

## 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 向量数据库 | Chroma |
| Embedding | BAAI/bge-large-zh-v1.5 |
| LLM 框架 | LangChain |
| 对话历史 | SQLite |

## 环境要求

- Python 3.10+
- PyTorch（支持 CPU / CUDA / MPS）

## 快速开始

### 本地运行

#### 1. 克隆项目

```bash
git clone https://github.com/niu0506/LawRAG.git
cd LawRAG
```

#### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

#### 3. 安装 PyTorch

根据你的系统选择合适的安装命令：

**Apple Silicon Mac (M系列芯片):**
```bash
pip install torch torchvision torchaudio
```

**Intel Mac / Windows CPU:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.4):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 4. 安装依赖

```bash
pip install -r requirements.txt
```

#### 5. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写必要的配置：

```env
# LLM 配置（必填）
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

# HuggingFace Token（可选，用于下载受限模型）
HF_TOKEN=
```

#### 6. 启动服务

```bash
python main.py
```

访问 http://localhost:8000 即可使用。

## 环境变量说明

| 变量 | 说明 | 必填 | 默认值 |
|------|------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | ✓ | - |
| `LLM_BASE_URL` | LLM API 地址 | ✓ | - |
| `LLM_MODEL` | 模型名称 | ✓ | - |
| `HF_TOKEN` | HuggingFace Token | | - |
| `EMBEDDING_MODEL` | Embedding 模型 | | BAAI/bge-large-zh-v1.5 |
| `CHUNK_SIZE` | 文本分块大小 | | 500 |
| `CHUNK_OVERLAP` | 分块重叠字符数 | | 50 |
| `TOP_K` | 检索返回文档数 | | 5 |
| `HISTORY_TURNS` | 多轮对话保留轮数 | | 5 |
| `HOST` | 服务监听地址 | | localhost |
| `PORT` | 服务监听端口 | | 8000 |
| `APP_PORT` | Docker 映射端口 | | 8000 |
| `MAX_UPLOAD_SIZE` | 文件上传大小限制 | | 50MB |

## API 文档

启动服务后访问 http://localhost:8000/docs 查看完整 API 文档。

### 问答接口

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

### 历史记录

```
GET    /api/history           # 会话列表
GET    /api/history/{id}      # 会话消息
DELETE /api/history/{id}      # 删除会话
DELETE /api/history           # 清空历史
```

### 系统状态

```
GET  /api/status    # 系统状态
GET  /api/health    # 健康检查
```

## 项目结构

```
LawRAG/
├── main.py              # FastAPI 服务入口
├── rag_engine.py        # RAG 引擎 + 对话历史管理
├── config.py            # 配置管理
├── index.html           # 前端页面
├── style.css            # 样式
├── script.js            # 前端脚本
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量示例
├── Dockerfile           # Docker 镜像配置
├── docker-compose.yml   # Docker Compose 配置
├── .dockerignore        # Docker 忽略文件
└── db/
    ├── chroma/          # 向量数据库存储
    └── history.db       # 对话历史数据库
```

## 使用说明

1. **上传法律文档**：点击页面上传按钮，选择 PDF 或 Word 格式的法律文档
2. **等待处理**：系统自动解析文档并建立向量索引
3. **开始提问**：在输入框中输入法律相关问题，系统会基于已上传的文档进行回答
4. **查看引用**：回答中会标注引用的法律条文来源

## 注意事项

- 首次运行时会自动下载 Embedding 模型（约 1.3GB），请确保网络通畅
- 上传的法律文档会按条文智能分割，保持法律条文的完整性
- 系统支持增量加载，已处理的文件不会重复处理
- 支持任意 OpenAI 兼容的 LLM API（如 DeepSeek、通义千问等）

## License

MIT
