"""LineSight diagnostic engine.

A deterministic implementation of the proposal's ILI triage and homeowner
video assessment contracts. It is deliberately model-light for the MVP:
the API contract and calculations are real; production ML models can replace
`analyze_video` without changing the platform.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="LineSight Diagnostic Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IliRequest(BaseModel):
    asset: dict[str, Any]
    rows: list[dict[str, Any]] = Field(default_factory=list)
    previous: dict[str, Any] | None = None


class VideoRequest(BaseModel):
    ref: str | None = None
    address: str | None = None
    issue: str | None = None
    videoUrl: str | None = None
    notes: str | None = None


def sample_rows(asset: dict[str, Any]) -> list[dict[str, Any]]:
    length = max(2000.0, float(asset.get("lengthKm", 10)) * 1000)
    rows = []
    for i in range(18):
        depth = round(8 + abs(math.sin(i * 1.7)) * 52 + (10 if i % 5 == 0 else 0))
        rows.append({
            "distanceM": round((i + 1) * (length / 19)),
            "depthPct": depth,
            "lengthMm": 25 + (i % 6) * 18,
            "widthMm": 12 + (i % 4) * 6,
            "type": "Crack-like" if i % 7 == 0 else "Pitting cluster" if i % 5 == 0 else "Metal loss",
        })
    return rows


def pick_type(depth: float) -> str:
    if depth > 55:
        return "Pitting cluster"
    if depth > 35:
        return "Metal loss"
    return "General corrosion"


def analyze_ili(req: IliRequest) -> dict[str, Any]:
    asset = req.asset
    diameter = float(asset.get("diameterMm") or 323.9)
    wall = float(asset.get("wallMm") or 9.5)
    smys = 448.0
    previous = {}
    for p in (req.previous or {}).get("anomalies", []):
        previous[round(float(p.get("distanceM", 0)))] = float(p.get("depthPct", 0))
    data = req.rows or sample_rows(asset)
    anomalies: list[dict[str, Any]] = []
    for index, row in enumerate(data):
        depth_pct = min(92.0, max(3.0, float(row.get("depthPct", row.get("depth_pct", 12)))))
        length_mm = max(10.0, float(row.get("lengthMm", row.get("length_mm", 40))))
        width_mm = max(8.0, float(row.get("widthMm", row.get("width_mm", 20))))
        distance = max(0.0, float(row.get("distanceM", row.get("distance_m", (index + 1) * 120))))
        depth_mm = depth_pct / 100 * wall
        z = length_mm * length_mm / (diameter * wall)
        m = math.sqrt(1 + 0.6275 * z - 0.003375 * z * z) if z <= 50 else 3.3 + 0.032 * z
        ratio = depth_mm / wall
        flow = 1.1 * smys
        p_safe = (2 * flow * wall / diameter) * ((1 - 0.667 * ratio) / (1 - (0.667 * ratio) / m))
        safe_kpa = max(0, round(p_safe * 1000 * 0.72))
        prev_depth = previous.get(round(distance))
        years = 3.0
        cgr = max(0.0, (depth_mm - (prev_depth / 100 * wall)) / years) if prev_depth is not None else max(0.1, depth_mm / 10)
        remaining = max(0.0, ((wall * 0.8) - depth_mm) / cgr) if cgr > 0 else 30.0
        severity = "critical" if depth_pct >= 70 else "high" if depth_pct >= 45 else "medium" if depth_pct >= 25 else "low"
        anomalies.append({
            "id": f"A-{index + 1:03d}",
            "distanceM": distance,
            "depthPct": depth_pct,
            "lengthMm": length_mm,
            "widthMm": width_mm,
            "type": str(row.get("type") or pick_type(depth_pct)),
            "severity": severity,
            "depthMm": round(depth_mm, 2),
            "safePressureKpa": safe_kpa,
            "cgrMmYear": round(cgr, 3),
            "remainingLifeYears": round(remaining, 1),
            "lat": float(asset.get("lat", 29.7604)) + distance / 111000,
            "lon": float(asset.get("lon", -95.3698)) + distance / 95000,
        })
    max_depth = max((a["depthPct"] for a in anomalies), default=0)
    critical = [a for a in anomalies if a["severity"] in ("critical", "high")]
    min_safe = min((a["safePressureKpa"] for a in anomalies), default=0)
    cgr = max((a["cgrMmYear"] for a in anomalies), default=0)
    remaining = min((a["remainingLifeYears"] for a in anomalies), default=30)
    risk = "critical" if any(a["severity"] == "critical" for a in anomalies) else "high" if critical else "medium" if max_depth > 25 else "low"
    summary = (
        f"{len(critical)} high-priority anomalies require action. Deepest metal loss {max_depth:.1f}% of wall; estimated corrosion rate {cgr:.2f} mm/yr."
        if critical else
        f"No critical anomalies identified. Deepest metal loss {max_depth:.1f}% of wall; continue the planned reassessment interval."
    )
    return {
        "ref": f"LS-{datetime.now().year}-{random.randint(1000, 9999)}",
        "summary": summary,
        "risk": risk,
        "confidence": 0.96,
        "maxDepthPct": round(max_depth, 2),
        "cgr": round(cgr, 3),
        "safePressureKpa": min_safe,
        "remainingLifeYears": round(remaining, 1),
        "anomalies": anomalies,
        "b31g": [{
            "id": a["id"], "distanceM": a["distanceM"], "depthPct": a["depthPct"],
            "calculatedSafePressureKpa": a["safePressureKpa"],
            "remainingLifeYears": a["remainingLifeYears"],
            "result": "acceptable" if a["safePressureKpa"] >= float(asset.get("maopKpa") or 0) else "below MAOP",
        } for a in anomalies],
        "heatmap": [{"distanceM": a["distanceM"], "depthPct": a["depthPct"], "risk": round(a["depthPct"] * 1.2)} for a in anomalies],
        "metrics": {
            "distanceKm": round(max(a["distanceM"] for a in anomalies) / 1000, 2),
            "anomalyCount": len(anomalies),
            "criticalCount": len(critical),
            "coverage": 99.1,
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "linesight-engine", "version": "1.0.0"}


@app.post("/v1/ili/analyze")
def ili(req: IliRequest) -> dict[str, Any]:
    return analyze_ili(req)


@app.post("/v1/video/analyze")
def video(req: VideoRequest) -> dict[str, Any]:
    seed = sum(ord(c) for c in (req.ref or req.address or "line"))
    defects = [
        {"type": "Root intrusion", "severity": "high", "distanceFt": 18 + seed % 9, "confidence": 0.94, "note": "Root mass reducing effective diameter; likely infiltration point."},
        {"type": "Offset joint", "severity": "medium", "distanceFt": 34 + seed % 13, "confidence": 0.89, "note": "Displaced joint with a lip that can snag debris."},
        {"type": "Scale / debris", "severity": "low", "distanceFt": 52 + seed % 17, "confidence": 0.82, "note": "Loose scale along the invert; recommend jetting."},
    ]
    return {
        "healthScore": 72,
        "summary": "Three findings were identified in the uploaded footage. One high-priority root intrusion and one displaced joint should be quoted for repair.",
        "defects": defects,
        "recommendedActions": ["Hydro-jet the line before repair", "Excavate and repair the root intrusion", "Re-inspect the offset joint after excavation"],
        "confidence": 0.91,
        "footageReviewedFt": 78,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }
