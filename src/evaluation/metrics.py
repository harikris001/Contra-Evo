"""Metrics logging to CSV and in-memory history."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

from utils.resources import ResourceStats


class MetricsLogger:
    """Logs per-generation metrics to CSV + JSON."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.json_path = self.run_dir / "metadata.json"
        self.history: List[Dict[str, Any]] = []
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames: List[str] | None = None

    def log(self, generation: int, stats: Dict[str, Any], resource: ResourceStats | None = None) -> None:
        row: Dict[str, Any] = {"generation": generation}
        row.update(stats)
        if resource:
            row.update(
                {
                    "ram_percent": resource.ram_percent,
                    "ram_used_mb": resource.ram_used_mb,
                    "cpu_percent": resource.cpu_percent,
                    "episodes_per_sec": resource.episodes_per_sec,
                    "emulator_fps": resource.emulator_fps,
                    "generation_duration": resource.generation_duration,
                }
            )
        self.history.append(row)
        # lazy init csv (append on resume so history isn't truncated)
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            append = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            self._csv_file = open(self.csv_path, "a" if append else "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            if not append:
                self._csv_writer.writeheader()
        else:
            # handle new fields
            for k in row.keys():
                if k not in self._fieldnames:  # type: ignore
                    # need to rewrite header? For simplicity, add to fieldnames and continue (may miss column)
                    self._fieldnames.append(k)  # type: ignore
        # fill missing
        for f in self._fieldnames:  # type: ignore
            if f not in row:
                row[f] = ""
        self._csv_writer.writerow(row)  # type: ignore
        self._csv_file.flush()  # type: ignore

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        with open(self.json_path, "w") as f:
            json.dump(meta, f, indent=2)

    def close(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def latest(self) -> Dict[str, Any] | None:
        return self.history[-1] if self.history else None
