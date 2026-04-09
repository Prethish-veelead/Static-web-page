# RAG-Powered AI Chatbot for Static Web Page

A production-ready React app with a RAG (Retrieval-Augmented Generation) chatbot that answers questions from a single uploaded help document (PDF). The app features a floating chat widget and is designed to deploy to Azure Static Web Apps.

## Architecture

```
FRONTEND (React 18 + Vite)
├── Chatbot Widget (floating button)
├── Chat Messages UI
└── Shopping List Feature

BACKEND (FastAPI)
├── /api/chat         → RAG pipeline + LLM response
├── /api/ingest       → PDF indexing
└── /api/health       → Health check

RAG PIPELINE
├── PDF Ingestion (PyMuPDF)
├── Text Chunking (LangChain)
├── Embeddings (OpenAI or HuggingFace)
├── Vector Search (FAISS)
└── LLM Generation (Claude or Azure OpenAI)
```

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.9+
- Your help document: `api/data/user_manual.pdf`

### 1. Clone and Install

```bash
git clone https://github.com/Prethish-veelead/Static-web-page.git
cd Static-web-page
```

### 2. Backend Setup

```bash
cd api
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill in your API keys:
- `ANTHROPIC_API_KEY` (if using Claude)
- Azure OpenAI credentials (if using Azure)

### 3. Place Your PDF

Copy your help document to:
```
api/data/user_manual.pdf
```

### 4. Ingest the PDF (one-time only)

```bash
# Start the backend
uvicorn main:app --reload --port 7071 &

# Ingest the PDF
curl -X POST http://localhost:7071/api/ingest
```

Response should show: `{"chunks_indexed": <number>}`

The FAISS index is saved to `api/data/faiss_index/` and reused across requests.

### 5. Frontend Setup

```bash
cd ../react-app
npm install
npm run dev
```

App runs at `http://localhost:5173`

- `/api` calls are proxied to `http://localhost:7071` (see `vite.config.ts`)
- Chatbot widget appears as a floating button (bottom-right)
- Click to toggle chat panel

## Project Structure

```
.
├── react-app/                    # Frontend (React 18 + Vite)
│   ├── src/
│   │   ├── components/
│   │   │   └── Chatbot/          # Chat widget component
│   │   ├── services/
│   │   │   └── chatService.ts    # API calls
│   │   ├── hooks/
│   │   │   └── useChat.ts        # Chat state hook
│   │   ├── types/
│   │   │   └── chat.ts           # TypeScript types
│   │   └── App.tsx               # Main app (includes Chatbot)
│   ├── vite.config.ts            # Vite config with /api proxy
│   ├── package.json
│   └── tsconfig.json
│
├── api/                          # Backend (FastAPI)
│   ├── main.py                   # FastAPI app + endpoints
│   ├── rag/
│   │   ├── ingest.py            # PDF → chunks → embeddings → FAISS
│   │   ├── retriever.py         # FAISS retrieval
│   │   ├── prompt_builder.py    # Prompt assembly
│   │   └── llm.py               # LLM generation (Anthropic/Azure)
│   ├── data/
│   │   ├── user_manual.pdf      # Your help document (place here)
│   │   └── faiss_index/         # Auto-generated FAISS index
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                     # (create from .env.example)
│
├── staticwebapp.config.json     # Azure Static Web Apps config
├── README.md                    # This file
└── .github/workflows/           # (optional) GitHub Actions CI/CD
```

## API Endpoints

### `POST /api/chat`
Chat with the RAG system.

**Request:**
```json
{
  "question": "How do I reset the device?",
  "history": [
    {"role": "user", "content": "What is this?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "answer": "To reset the device, press and hold the power button for 10 seconds. [Page 42]",
  "sources": ["Page 42", "Page 85"]
}
```

### `POST /api/ingest`
Index the PDF once.

**Response:**
```json
{
  "chunks_indexed": 2400
}
```

### `GET /api/health`
Health check.

**Response:**
```json
{
  "status": "ok"
}
```

## Frontend Components

### Chatbot Widget

Located at `react-app/src/components/Chatbot/Chatbot.tsx`

- **Floating bubble:** 56px circle, bottom-right corner, brand blue (#1565C0)
- **Chat panel:** 360px × 520px, white background
- **Mobile:** Full-screen on viewports < 480px
- **Features:**
  - Message history (scrollable)
  - Typing indicator while loading
  - Clear chat button
  - Source citations (page numbers)
  - User messages (right, blue) vs. Bot messages (left, gray)

### Chat Service

Located at `react-app/src/services/chatService.ts`

- Sends `POST /api/chat` requests
- Handles network errors gracefully
- Returns typed responses or error messages

### useChat Hook

Located at `react-app/src/hooks/useChat.ts`

- Manages messages state
- Handles loading state
- Integrates with chatService
- Exposes: `messages`, `isLoading`, `sendMessage()`, `clearChat()`

## Backend Configuration

### Environment Variables

Create `api/.env` (copy from `.env.example`):

```env
# LLM Provider: "anthropic" or "azure_openai"
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Embeddings: "openai" or "huggingface"
EMBEDDING_PROVIDER=openai

# Azure OpenAI (if using)
AZURE_OPENAI_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...

# CORS origins
CORS_ORIGINS=http://localhost:5173,https://your-app.azurestaticapps.net
```

### RAG Pipeline

1. **Ingest (`api/rag/ingest.py`)**
   - Extracts text page-by-page from PDF
   - Chunks with LangChain (`chunk_size=800`, `chunk_overlap=100`)
   - Embeds with OpenAI or HuggingFace
   - Saves FAISS index to disk

2. **Retrieval (`api/rag/retriever.py`)**
   - Loads FAISS index at startup (singleton)
   - Embeds query, finds top-5 similar chunks
   - Returns chunks with page metadata

3. **Prompt Building (`api/rag/prompt_builder.py`)**
   - Assemble system prompt (strict RAG constraints)
   - Format context from retrieved chunks
   - Include conversation history

4. **LLM (`api/rag/llm.py`)**
   - **Anthropic:** Claude Sonnet 4.6, `temperature=0`, max_tokens=1024
   - **Azure OpenAI:** GPT-4o, same params
   - System prompt enforces "answer only from context"

## Deployment to Azure Static Web Apps

### Option A: Azure Functions (Recommended)

1. Create Azure Functions app in Python v2 model
2. Migrate endpoints to Azure Functions:
   - `functions/chat/__init__.py` → `POST /api/chat`
   - `functions/ingest/__init__.py` → `POST /api/ingest`
   - `functions/health/__init__.py` → `GET /api/health`
3. Package FAISS index with function app
4. Update GitHub Actions YAML

### Option B: Azure Container App

1. Create Dockerfile in `api/`
2. Deploy FastAPI app as Azure Container App
3. Configure custom backend URL in Static Web Apps

### GitHub Actions

Update `.github/workflows/` with:

```yaml
- name: Build And Deploy
  uses: Azure/static-web-apps-deploy@v1
  with:
    app_location: "react-app"
    api_location: "functions"  # if using Option A
    output_location: "dist"
```

## Styling

- **CSS Modules** only (no Tailwind, no MUI)
- **Chatbot colors:**
  - Primary: `#1565c0` (brand blue)
  - Messages: white background, shadows
  - Text: dark gray on light backgrounds
- **Responsive:** Mobile panel goes full-screen on < 480px

## Quality Requirements

- ✅ Chatbot answers ONLY from PDF (enforced in system prompt, `temperature=0`)
- ✅ Every answer cites page numbers
- ✅ FAISS index loaded once at startup
- ✅ CORS configured (React ↔ Backend)
- ✅ No `any` types in React code (TypeScript strict)
- ✅ Network errors handled gracefully
- ✅ Mobile-responsive chat UI

## Troubleshooting

### "Run POST /api/ingest first"
The FAISS index hasn't been created yet. Make sure `api/data/user_manual.pdf` exists, then:
```bash
curl -X POST http://localhost:7071/api/ingest
```

### "PDF not found"
Ensure the file exists at `api/data/user_manual.pdf`

### CORS errors
Check `CORS_ORIGINS` in `.env` includes `http://localhost:5173` (or your frontend URL)

### Embedding API errors
- For OpenAI: check `ANTHROPIC_API_KEY` or `AZURE_OPENAI_KEY`
- For HuggingFace: embeddings download automatically (requires internet)

### "No such table" or database errors
This app doesn't use a database. All data is in-memory or on disk (FAISS index).

## Development

### Adding new endpoints
1. Add route in `api/main.py`
2. Use RAG retriever: `from rag.retriever import RAGRetriever`
3. Call LLM: `from rag.llm import generate_answer`

### Updating chatbot UI
1. Edit `react-app/src/components/Chatbot/Chatbot.tsx`
2. Styles in `react-app/src/components/Chatbot/Chatbot.module.css`
3. Auto-reload on save

### Testing
```bash
# Backend health check
curl http://localhost:7071/api/health

# Chat request
curl -X POST http://localhost:7071/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this?", "history": []}'
```

## License

This project is provided as-is for use with Azure Static Web Apps.

## Support

For issues or questions, refer to:
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [React Docs](https://react.dev/)
- [Azure Static Web Apps Docs](https://learn.microsoft.com/azure/static-web-apps/)