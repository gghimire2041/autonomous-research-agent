Autonomous Research Agent
A production-grade autonomous research agent with safety guardrails, tool integration, and comprehensive observability.
📁 Repository Structure
autonomous-research-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # API endpoints
│   │   └── models.py              # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py               # Main agent logic
│   │   ├── planner.py             # Task planning strategies
│   │   ├── memory.py              # SQLite memory store
│   │   └── guardrails.py          # Safety mechanisms
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                # Tool interface
│   │   ├── web_search.py          # DuckDuckGo search
│   │   ├── web_fetch.py           # Web content fetching
│   │   ├── calculator.py          # Math operations
│   │   ├── file_ops.py            # File read/write
│   │   └── sql_query.py           # SQLite queries
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── adapters.py            # OpenAI/HF adapters
│   │   └── prompts.py             # Prompt templates
│   └── utils/
│       ├── __init__.py
│       ├── security.py            # Security utilities
│       ├── observability.py       # Logging/metrics
│       └── config.py              # Settings management
├── cli/
│   ├── __init__.py
│   └── client.py                  # CLI interface
├── ui/
│   ├── package.json
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── TaskForm.js
│   │   │   └── TaskTrace.js
│   │   └── index.js
│   └── public/
│       └── index.html
├── eval/
│   ├── __init__.py
│   ├── benchmarks.py              # Synthetic benchmarks
│   ├── scoring.py                 # Evaluation metrics
│   └── tasks/
│       ├── fact_collection.py
│       └── web_research.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_agent.py
│   ├── test_tools.py
│   ├── test_api.py
│   └── test_security.py
├── docs/
│   ├── README.md
│   ├── THREAT_MODEL.md
│   ├── SAFETY_GUIDELINES.md
│   └── diagrams/
│       ├── agent_loop.mmd
│       └── tool_calls.mmd
├── scripts/
│   ├── setup.sh
│   └── demo.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── Dockerfile
├── LICENSE
├── Makefile
├── pyproject.toml
└── README.md
🚀 Quick Start
bash# Setup
make setup

# Run development server
make run

# Run demo
make demo

# Run tests
make test
🔧 Configuration
Copy .env.example to .env and configure:
bash# LLM Configuration
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
MODEL_NAME=gpt-4-turbo-preview

# Safety Settings
SAFE_MODE=true
MAX_STEPS=10
TIMEOUT_SECONDS=300
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Database
DATABASE_URL=sqlite:///./agent_memory.db

# Observability
LOG_LEVEL=INFO
ENABLE_METRICS=true
ENABLE_TRACING=true
🛡️ Safety Features

Guardrails: URL allowlists, prompt injection detection
Sandboxing: File operations restricted to safe directories
Rate Limiting: Configurable per-tool limits
Timeouts: Maximum execution time per task
PII Redaction: Automatic scrubbing of sensitive data

📊 Observability

Structured Logging: JSON logs with trace correlation
Metrics: Prometheus-compatible metrics
Tracing: OpenTelemetry distributed tracing
Health Checks: Endpoint monitoring

🎯 Evaluation
Built-in evaluation framework with:

Synthetic benchmarks
Golden answer comparisons
Performance metrics (latency, accuracy, step count)
Automated scoring

🤝 Contributing

Install pre-commit hooks: pre-commit install
Run tests: make test
Follow conventional commits
Ensure 100% type coverage with mypy

📜 License
MIT License - see LICENSE file.
