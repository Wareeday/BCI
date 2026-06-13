"""
ml/model_registry.py
=====================
ML model registry — tracks versions, accuracy, and deployment status.

Ensures the correct model version is loaded for each user session.
Supports rollback if a new model degrades accuracy.

FDA 21 CFR Part 11: model changes must be logged in the audit trail.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from loguru import logger


@dataclass
class ModelEntry:
    """One registered model version."""
    model_id: str
    model_name: str           # cnn_p300 | cnn_motor_imagery | eegnet | lda
    version: str
    file_path: str
    accuracy: Optional[float]
    inference_ms: Optional[float]
    training_samples: Optional[int]
    dataset: str
    deployed: bool
    created_at: float = field(default_factory=time.time)
    checksum_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelRegistry:
    """
    File-backed model registry.
    Persists to ml/saved_models/registry.json.
    """

    REGISTRY_FILE = "ml/saved_models/registry.json"

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}
        self._load()

    def register(
        self,
        model_name: str,
        version: str,
        file_path: str,
        accuracy: Optional[float] = None,
        inference_ms: Optional[float] = None,
        training_samples: Optional[int] = None,
        dataset: str = "BCI Competition IV 2a",
        deploy: bool = False,
    ) -> ModelEntry:
        """Register a new model version."""
        model_id = f"{model_name}_v{version}"
        checksum = self._sha256(file_path) if os.path.exists(file_path) else None

        entry = ModelEntry(
            model_id=model_id,
            model_name=model_name,
            version=version,
            file_path=file_path,
            accuracy=accuracy,
            inference_ms=inference_ms,
            training_samples=training_samples,
            dataset=dataset,
            deployed=deploy,
            checksum_sha256=checksum,
        )
        self._entries[model_id] = entry
        if deploy:
            self._undeploy_others(model_name)
            entry.deployed = True

        self._save()
        logger.info(
            f"Model registered: {model_id}, acc={accuracy}, "
            f"deployed={deploy}"
        )
        return entry

    def get_deployed(self, model_name: str) -> Optional[ModelEntry]:
        """Return the currently deployed version of a model."""
        for entry in self._entries.values():
            if entry.model_name == model_name and entry.deployed:
                return entry
        return None

    def list_all(self, model_name: Optional[str] = None) -> list[ModelEntry]:
        entries = list(self._entries.values())
        if model_name:
            entries = [e for e in entries if e.model_name == model_name]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def rollback(self, model_name: str) -> Optional[ModelEntry]:
        """Deploy the previous version of a model."""
        versions = [
            e for e in self._entries.values()
            if e.model_name == model_name
        ]
        versions.sort(key=lambda e: e.created_at, reverse=True)
        if len(versions) < 2:
            logger.warning(f"No previous version to rollback to for {model_name}")
            return None
        self._undeploy_others(model_name)
        versions[1].deployed = True
        self._save()
        logger.warning(f"Rollback: {model_name} → {versions[1].version}")
        return versions[1]

    def _undeploy_others(self, model_name: str):
        for entry in self._entries.values():
            if entry.model_name == model_name:
                entry.deployed = False

    def _sha256(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _save(self):
        Path(self.REGISTRY_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(self.REGISTRY_FILE, "w") as f:
            json.dump([e.to_dict() for e in self._entries.values()], f, indent=2)

    def _load(self):
        if not os.path.exists(self.REGISTRY_FILE):
            return
        try:
            with open(self.REGISTRY_FILE) as f:
                data = json.load(f)
            for d in data:
                entry = ModelEntry(**d)
                self._entries[entry.model_id] = entry
        except Exception as exc:
            logger.warning(f"Could not load model registry: {exc}")


# Singleton
registry = ModelRegistry()