How to run?
mongodb
source .venv/bin/activate
uvicorn auth_service:app --host 127.0.0.1 --port 9000 --reload

backend
source .venv/bin/activate
uvicorn app:app --reload --port 8000

frontend
python3 -m http.server 5173
Serving HTTP on :: port 5173 (http://[::]:5173/)




Tech stack at a glance
    •    FastAPI (+ CORS) for HTTP endpoints
    •    SentenceTransformers (SBERT) for multilingual embeddings
    •    FAISS (optional) for fast ANN search
    •    LogisticRegression + simple heuristics for ingredient category detection
    •    NumPy for vectors
    •    ReportLab (optional) for nice PDFs; plain text fallback
    •    Azure OpenAI (optional) for LLM advice
    •    .env config via python-dotenv
    •    File caching under .cache/ for embeddings and FAISS index
Core
    •    Language: Python 3
    •    Web framework: FastAPI
    •    Server (run-time): Uvicorn (ASGI)

Data & Models
    •    Embeddings / NLP: sentence-transformers (SBERT; default paraphrase-multilingual-MiniLM-L12-v2)
    •    Vector search: FAISS (optional; falls back to NumPy dot products)
    •    Classical ML: scikit-learn (LogisticRegression, LabelEncoder)
    •    Numerics: NumPy
    •    Validation / Schemas: Pydantic (BaseModel)

AI Integration 
    •    LLM client: Azure OpenAI (openai Azure SDK)
Env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_VERSION, AZURE_OPENAI_DEPLOYMENT

I/O & Data Sources
    •    Recipes: Local JSON files in recipes_data/…
    •    Overrides: CSV (database.csv) for intensities & seasonality (auto-detected columns)
    •    Env config: python-dotenv (.env)

API Responses & Middleware
    •    CORS: fastapi.middleware.cors.CORSMiddleware
    •    Streaming files: fastapi.responses.StreamingResponse

PDFs (nice-to-have)
    •    Primary: ReportLab (pretty A4 PDFs)
    •    Fallback: Plain-text PDF bytes if ReportLab missing

Caching & Performance
    •    Function-level cache: functools.lru_cache
    •    On-disk cache: .cache/ for embeddings & metadata
    •    Warmup thread: threading to prebuild embeddings/FAISS

Utilities
    •    Typing: typing (Optional, List, Tuple, Dict, etc.)
    •    Paths: pathlib.Path
    •    Dates: datetime
    •    Regex/HTML: re, html.unescape

Dev/Deploy notes
    •    Run: uvicorn <module>:app --reload
    •    Optional deps: faiss-cpu, reportlab
    •    Must-have deps: fastapi, uvicorn, numpy, scikit-learn, sentence-transformers, python-dotenv, pydantic, openai 



<img width="2150" height="1312" alt="image" src="https://github.com/user-attachments/assets/9426ec9d-ac9c-430e-9cb3-16c77996b87b" />



<img width="2306" height="1312" alt="image" src="https://github.com/user-attachments/assets/0f9d8bc0-afe6-4e4c-8cc1-20e96cdd5466" />



<img width="2306" height="1312" alt="image" src="https://github.com/user-attachments/assets/33c4ba6a-bb77-4d90-8690-d629463415f3" />



<img width="772" height="1366" alt="image" src="https://github.com/user-attachments/assets/39f69772-f799-45a6-9408-70a4352aebe4" />



<img width="772" height="1366" alt="image" src="https://github.com/user-attachments/assets/35a62863-54c0-4f66-bf62-43132c7d87b4" />



