"""
MLflow Service — tracks every extraction as an experiment run.
Enables prompt versioning, A/B comparison, and drift detection.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MLflowService:
    """Manages MLflow experiment tracking for extraction runs."""

    def __init__(self):
        self._connected = False
        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow connection and experiment."""
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            self._connected = True
            logger.info(
                "mlflow_connected",
                uri=settings.mlflow_tracking_uri,
                experiment=settings.mlflow_experiment_name,
            )
        except Exception as e:
            logger.warning("mlflow_connection_failed", error=str(e))
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def log_extraction_run(
        self,
        document_id: str,
        document_type: str,
        prompt_version: str,
        ocr_confidence: float,
        extraction_confidence: float,
        fields_extracted: int,
        total_fields: int,
        processing_time_ms: float,
        ocr_engine: str,
        llm_model: str,
        warnings: list[str] | None = None,
    ) -> str | None:
        """
        Log an extraction run to MLflow with all relevant metrics.
        Returns the MLflow run ID.
        """
        if not self._connected:
            return None

        try:
            import mlflow

            with mlflow.start_run() as run:
                # Parameters — what configuration was used
                mlflow.log_param("document_id", document_id)
                mlflow.log_param("document_type", document_type)
                mlflow.log_param("prompt_version", prompt_version)
                mlflow.log_param("ocr_engine", ocr_engine)
                mlflow.log_param("llm_model", llm_model)

                # Metrics — how well did it perform
                mlflow.log_metric("ocr_confidence", ocr_confidence)
                mlflow.log_metric("extraction_confidence", extraction_confidence)
                mlflow.log_metric("fields_extracted", fields_extracted)
                mlflow.log_metric("total_fields", total_fields)
                mlflow.log_metric(
                    "field_completeness",
                    fields_extracted / max(total_fields, 1),
                )
                mlflow.log_metric("processing_time_ms", processing_time_ms)

                # Tags
                mlflow.set_tag("pipeline", "docai-extraction")
                mlflow.set_tag("environment", "production")
                if warnings:
                    mlflow.set_tag("has_warnings", "true")
                    mlflow.set_tag("warnings", "; ".join(warnings[:5]))

                run_id = run.info.run_id

            logger.info("mlflow_run_logged", run_id=run_id, doc_id=document_id)
            return run_id

        except Exception as e:
            logger.error("mlflow_logging_failed", error=str(e))
            return None

    async def get_prompt_comparison(
        self, document_type: str
    ) -> dict[str, Any]:
        """
        Compare prompt version performance for A/B testing.
        Returns metrics grouped by prompt version.
        """
        if not self._connected:
            return {}

        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(
                settings.mlflow_experiment_name
            )
            if not experiment:
                return {}

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.document_type = '{document_type}'",
                max_results=500,
            )

            if runs.empty:
                return {}

            # Group by prompt version
            comparison = {}
            for version, group in runs.groupby("params.prompt_version"):
                comparison[version] = {
                    "run_count": len(group),
                    "avg_extraction_confidence": round(
                        group["metrics.extraction_confidence"].mean(), 4
                    ),
                    "avg_processing_time_ms": round(
                        group["metrics.processing_time_ms"].mean(), 2
                    ),
                    "avg_field_completeness": round(
                        group["metrics.field_completeness"].mean(), 4
                    ),
                }

            return comparison

        except Exception as e:
            logger.error("prompt_comparison_failed", error=str(e))
            return {}

    async def check_drift(self, document_type: str, window: int = 50) -> dict[str, Any]:
        """
        Check for extraction quality drift by comparing recent runs
        against historical baseline.
        """
        if not self._connected:
            return {"status": "mlflow_unavailable"}

        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(
                settings.mlflow_experiment_name
            )
            if not experiment:
                return {"status": "no_data"}

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.document_type = '{document_type}'",
                order_by=["start_time DESC"],
                max_results=window * 2,
            )

            if len(runs) < window:
                return {"status": "insufficient_data", "runs": len(runs)}

            recent = runs.head(window)
            baseline = runs.tail(window)

            recent_conf = recent["metrics.extraction_confidence"].mean()
            baseline_conf = baseline["metrics.extraction_confidence"].mean()
            drift = baseline_conf - recent_conf

            return {
                "status": "drift_detected" if drift > 0.05 else "stable",
                "recent_avg_confidence": round(recent_conf, 4),
                "baseline_avg_confidence": round(baseline_conf, 4),
                "drift_magnitude": round(drift, 4),
                "threshold": settings.accuracy_threshold,
                "recommendation": (
                    "Consider retraining or updating prompts"
                    if drift > 0.05
                    else "No action needed"
                ),
            }

        except Exception as e:
            logger.error("drift_check_failed", error=str(e))
            return {"status": "error", "error": str(e)}


# Singleton
_mlflow_service: MLflowService | None = None


def get_mlflow_service() -> MLflowService:
    global _mlflow_service
    if _mlflow_service is None:
        _mlflow_service = MLflowService()
    return _mlflow_service
