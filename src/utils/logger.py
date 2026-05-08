"""CSV logger for training metrics."""

from __future__ import annotations

import csv
import os
from pathlib import Path


class CSVLogger:
    """Append-mode CSV logger for training metrics."""

    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        """
        Initialise logger.

        Args:
            path: File path for the CSV output.
            fieldnames: Column names for the CSV header.
        """
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)

        write_header = not self.path.exists() or os.path.getsize(self.path) == 0
        self._file = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if write_header:
            self._writer.writeheader()

    def log(self, row: dict) -> None:
        """Append a row to the CSV file."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        self._file.flush()
        self._file.close()

    def __enter__(self) -> "CSVLogger":
        """Support use as a context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close on context exit."""
        self.close()
