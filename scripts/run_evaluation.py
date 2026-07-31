from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from evaluation.runner import load_cases, run_evaluation


class HttpCopilotClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def ask(self, *, session_id: str, message: str) -> dict:
        response = httpx.post(
            f"{self._base_url}/copilot",
            json={"session_id": session_id, "message": message},
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic evaluations against the copilot API."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="evaluation/dataset.json")
    parser.add_argument("--output", default="evaluation/results/latest.json")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    report = run_evaluation(
        client=HttpCopilotClient(args.api_url, args.timeout_seconds),
        cases=cases,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = report.to_dict()["summary"]
    print("Evaluation Summary")
    print("------------------")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}")
    print(f"Failed: {summary['failed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    print(f"Average duration: {summary['average_duration_ms']:.2f} ms")
    print(f"Average tokens: {summary['average_total_tokens']:.2f}")
    print(f"Report: {output}")
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
