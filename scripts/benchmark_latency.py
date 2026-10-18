"""Benchmark script measuring PredictIQ API latency and throughput.

Evaluates local prediction latency with representative customer payloads across
warmup, single-record scoring, and varying batch sizes. Reports actual p50 and p95 latencies.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

from starlette.testclient import TestClient

from predictiq.api.main import app

SAMPLE_RECORD: Dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.80,
    "TotalCharges": 1017.60,
}


def benchmark_endpoint(
    client: TestClient,
    endpoint: str,
    payload: Dict[str, Any],
    n_iterations: int = 100,
    n_warmup: int = 20,
) -> Dict[str, float]:
    """Run warm-up and timed iterations against a given endpoint."""
    # Warm-up phase
    for _ in range(n_warmup):
        res = client.post(endpoint, json=payload)
        if res.status_code != 200:
            raise RuntimeError(f"Warm-up failed with status {res.status_code}: {res.text}")

    # Timing phase
    durations_ms: List[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        res = client.post(endpoint, json=payload)
        t1 = time.perf_counter()
        if res.status_code != 200:
            raise RuntimeError(f"Benchmark iteration failed with status {res.status_code}: {res.text}")
        durations_ms.append((t1 - t0) * 1000.0)

    durations_ms.sort()
    p50 = statistics.median(durations_ms)
    p95_idx = int(0.95 * len(durations_ms))
    p95 = durations_ms[p95_idx]
    mean = statistics.mean(durations_ms)
    min_lat = min(durations_ms)
    max_lat = max(durations_ms)

    return {
        "iterations": n_iterations,
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
    }


def main() -> None:
    print("=" * 70)
    print("PredictIQ REST API Latency Benchmark")
    print("=" * 70)
    print("Initializing FastAPI TestClient with loaded champion model...")

    with TestClient(app) as client:
        # Verify readiness
        ready_res = client.get("/ready")
        assert ready_res.status_code == 200, f"API is not ready: {ready_res.text}"
        model_name = ready_res.json().get("model_name", "Unknown")
        print(f"Target Model: {model_name}\n")

        # 1. Single Record /predict
        print("Benchmarking single-record POST /predict (100 iterations)...")
        single_metrics = benchmark_endpoint(
            client=client,
            endpoint="/predict",
            payload=SAMPLE_RECORD,
            n_iterations=100,
            n_warmup=20,
        )
        print(f"  -> Median (p50): {single_metrics['p50_ms']} ms")
        print(f"  -> 95th Percentile (p95): {single_metrics['p95_ms']} ms")
        print(f"  -> Mean: {single_metrics['mean_ms']} ms | Range: [{single_metrics['min_ms']} - {single_metrics['max_ms']}] ms\n")

        # 2. Batch of 10 records
        print("Benchmarking batch POST /predict/batch (batch_size=10, 50 iterations)...")
        batch_10_payload = {"records": [SAMPLE_RECORD] * 10}
        batch_10_metrics = benchmark_endpoint(
            client=client,
            endpoint="/predict/batch",
            payload=batch_10_payload,
            n_iterations=50,
            n_warmup=10,
        )
        print(f"  -> Median (p50): {batch_10_metrics['p50_ms']} ms ({batch_10_metrics['p50_ms']/10:.2f} ms/record)")
        print(f"  -> 95th Percentile (p95): {batch_10_metrics['p95_ms']} ms")
        print(f"  -> Mean: {batch_10_metrics['mean_ms']} ms | Range: [{batch_10_metrics['min_ms']} - {batch_10_metrics['max_ms']}] ms\n")

        # 3. Batch of 50 records
        print("Benchmarking batch POST /predict/batch (batch_size=50, 25 iterations)...")
        batch_50_payload = {"records": [SAMPLE_RECORD] * 50}
        batch_50_metrics = benchmark_endpoint(
            client=client,
            endpoint="/predict/batch",
            payload=batch_50_payload,
            n_iterations=25,
            n_warmup=5,
        )
        print(f"  -> Median (p50): {batch_50_metrics['p50_ms']} ms ({batch_50_metrics['p50_ms']/50:.2f} ms/record)")
        print(f"  -> 95th Percentile (p95): {batch_50_metrics['p95_ms']} ms")
        print(f"  -> Mean: {batch_50_metrics['mean_ms']} ms | Range: [{batch_50_metrics['min_ms']} - {batch_50_metrics['max_ms']}] ms\n")

    print("=" * 70)
    print("Benchmark Completed Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
