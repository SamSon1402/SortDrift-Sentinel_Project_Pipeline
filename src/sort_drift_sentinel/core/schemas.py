from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class FrameQuality(FlexibleModel):
    brightness: float = 128.0
    blur_variance: float = 100.0
    valid: bool = True
    reasons: list[str] = Field(default_factory=list)


class NIRReading(FlexibleModel):
    polyester: float = 0.25
    cotton: float = 0.50
    wool: float = 0.10
    other: float = 0.15
    signal_quality: float = 1.0


class Defect(FlexibleModel):
    kind: str = "unknown"
    confidence: float = Field(default=0.0, ge=0, le=1)
    severity: float = Field(default=0.0, ge=0, le=1)


class VisionResult(FlexibleModel):
    category: str = "unknown"
    category_confidence: float = Field(default=0.5, ge=0, le=1)
    brand_top5: list[tuple[str, float]] = Field(default_factory=list)
    attributes: dict[str, tuple[str, float]] = Field(default_factory=dict)
    defects: list[Defect] = Field(default_factory=list)
    embedding_norm: float = 1.0
    ood_score: float = Field(default=0.0, ge=0, le=1)
    model_version: str = "unknown"
    inference_ms: float = 0.0


class GradeDecision(FlexibleModel):
    grade: str = "REVIEW"
    confidence: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""


class RouteDecision(FlexibleModel):
    route: str = "manual_review"
    reason: str = ""
    requires_human: bool = False


class GarmentEvent(FlexibleModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    garment_id: str
    customer: str
    policy_version: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    frame_quality: FrameQuality = Field(default_factory=FrameQuality)
    nir: NIRReading = Field(default_factory=NIRReading)
    vision: VisionResult = Field(default_factory=VisionResult)
    grade: GradeDecision = Field(default_factory=GradeDecision)
    route: RouteDecision = Field(default_factory=RouteDecision)
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def deployment_id(self) -> str:
        return str(self.metadata.get("deployment_id") or f"{self.customer}-default")

    @property
    def camera_id(self) -> str:
        return str(self.metadata.get("camera_id") or "camera-unknown")

    @property
    def taxonomy_version(self) -> str:
        return str(self.metadata.get("taxonomy_version") or f"policy-{self.policy_version}")

    @property
    def embedding(self) -> list[float] | None:
        value = self.metadata.get("embedding")
        if isinstance(value, list) and value and all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]
        return None

    @property
    def min_task_confidence(self) -> float:
        values = [self.vision.category_confidence, self.grade.confidence]
        if self.vision.brand_top5:
            values.append(float(self.vision.brand_top5[0][1]))
        values.extend(float(score) for _, score in self.vision.attributes.values())
        values.extend(float(d.confidence) for d in self.vision.defects)
        return max(0.0, min(1.0, min(values))) if values else 0.5

    @property
    def max_defect_severity(self) -> float:
        return max((d.severity for d in self.vision.defects), default=0.0)


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    WARMING_UP = "WARMING_UP"


class DriftMetric(BaseModel):
    name: str
    value: float
    warning_threshold: float
    critical_threshold: float
    status: HealthStatus


class DriftReport(BaseModel):
    deployment_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sample_count: int
    input_health: HealthStatus
    model_health: HealthStatus
    runtime_health: HealthStatus
    business_health: HealthStatus
    overall: HealthStatus
    metrics: dict[str, float]
    likely_cause: str
    evidence: list[str]
    recommended_action: str


class IncidentSeverity(StrEnum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Incident(BaseModel):
    incident_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    severity: IncidentSeverity
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    likely_cause: str
    evidence: list[str]
    recommended_action: str
    model_version: str | None = None
    camera_id: str | None = None
    status: str = "OPEN"


class ActiveLearningItem(BaseModel):
    event_id: str
    garment_id: str
    deployment_id: str
    priority: float
    uncertainty: float
    ood: float
    business_risk: float
    diversity: float
    reason: str


class ReleaseMetrics(BaseModel):
    defect_recall: float = Field(ge=0, le=1)
    grade_macro_f1: float = Field(ge=0, le=1)
    route_agreement: float = Field(ge=0, le=1)
    p99_latency_ms: float = Field(gt=0)
    manual_review_rate: float = Field(ge=0, le=1)


class ReleaseState(StrEnum):
    REGISTERED = "REGISTERED"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"


class ReleaseRecord(BaseModel):
    release_id: str = Field(default_factory=lambda: str(uuid4()))
    deployment_id: str
    current_model: str
    candidate_model: str
    current_metrics: ReleaseMetrics
    candidate_metrics: ReleaseMetrics
    passed_gate: bool
    gate_reasons: list[str]
    state: ReleaseState
    canary_percent: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
