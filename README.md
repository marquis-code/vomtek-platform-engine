# LineSight Diagnostic Engine

FastAPI compute service for ILI screening and homeowner visual triage.

## Run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 4210
```

## Endpoints
- `GET /health`
- `POST /v1/ili/analyze` — anomaly rows + asset → B31G, CGR, remaining life,
  safe pressure, heatmap and summary.
- `POST /v1/video/analyze` — deterministic visual findings behind a stable
  contract.

The API calls this service and falls back locally if it is unavailable. The
video scorer is a demo implementation; production ML should replace
`video()` while preserving the response contract.
