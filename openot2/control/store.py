"""Simple JSON-file persistence for task runs and events."""

from __future__ import annotations

from pathlib import Path

from .models import RunEvent, RunSequence, TaskRun


class JsonRunStore:
    """Persist :class:`TaskRun` objects as JSON files and
    :class:`RunEvent` objects as JSON-Lines files.

    Directory layout::

        base_dir/
          runs/{run_id}.json
          events/{run_id}.jsonl
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self._runs_dir = self.base_dir / "runs"
        self._events_dir = self.base_dir / "events"
        self._sequences_dir = self.base_dir / "sequences"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._sequences_dir.mkdir(parents=True, exist_ok=True)

    # -- runs ---------------------------------------------------------------

    def _run_path(self, run_id: str) -> Path:
        return self._runs_dir / f"{run_id}.json"

    def create_run(self, run: TaskRun) -> TaskRun:
        """Save a new run to disk and return it."""
        path = self._run_path(run.id)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        return run

    def save_run(self, run: TaskRun) -> None:
        """Overwrite an existing run file on disk."""
        path = self._run_path(run.id)
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def load_run(self, run_id: str) -> TaskRun:
        """Load a run from its JSON file."""
        path = self._run_path(run_id)
        return TaskRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[TaskRun]:
        """Load every run stored on disk."""
        runs: list[TaskRun] = []
        for path in sorted(self._runs_dir.glob("*.json")):
            runs.append(TaskRun.model_validate_json(path.read_text(encoding="utf-8")))
        return runs

    # -- events -------------------------------------------------------------

    def _events_path(self, run_id: str) -> Path:
        return self._events_dir / f"{run_id}.jsonl"

    def append_event(self, event: RunEvent) -> None:
        """Append a single event to the JSONL log for its run."""
        path = self._events_path(event.run_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")

    def list_events(self, run_id: str) -> list[RunEvent]:
        """Read all events for a given run."""
        path = self._events_path(run_id)
        if not path.exists():
            return []
        events: list[RunEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(RunEvent.model_validate_json(line))
        return events

    # -- sequences ----------------------------------------------------------

    def _sequence_path(self, seq_id: str) -> Path:
        return self._sequences_dir / f"{seq_id}.json"

    def create_sequence(self, seq: RunSequence) -> RunSequence:
        """Save a new sequence to disk and return it."""
        path = self._sequence_path(seq.id)
        path.write_text(seq.model_dump_json(indent=2), encoding="utf-8")
        return seq

    def load_sequence(self, seq_id: str) -> RunSequence:
        """Load a sequence from its JSON file."""
        path = self._sequence_path(seq_id)
        return RunSequence.model_validate_json(path.read_text(encoding="utf-8"))

    def save_sequence(self, seq: RunSequence) -> None:
        """Overwrite an existing sequence file on disk."""
        path = self._sequence_path(seq.id)
        path.write_text(seq.model_dump_json(indent=2), encoding="utf-8")

    def list_sequences(self) -> list[RunSequence]:
        """Load every sequence stored on disk."""
        sequences: list[RunSequence] = []
        for path in sorted(self._sequences_dir.glob("*.json")):
            sequences.append(RunSequence.model_validate_json(path.read_text(encoding="utf-8")))
        return sequences
