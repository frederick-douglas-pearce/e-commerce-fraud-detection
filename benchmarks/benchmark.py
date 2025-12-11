#!/usr/bin/env python3
"""
E-Commerce Fraud Detection API Benchmarking Script

Comprehensive performance testing suite that measures:
- Single request latency (P50, P95, P99)
- Concurrent request throughput
- Server-side vs end-to-end timing
- Cold start performance

Usage:
    python benchmark.py --url http://localhost:8000 --iterations 100 --concurrent 10

    # Docker deployment
    python benchmark.py --url http://localhost:8000

    # Local deployment
    python benchmark.py --url http://localhost:8000 --label "local"

Requirements:
    pip install requests
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests


class FraudAPIBenchmark:
    """Benchmark suite for Fraud Detection API."""

    def __init__(self, base_url: str, iterations: int = 100, concurrent_users: int = 10, output_path: str = "benchmark_results.json", with_explanation: bool = False):
        self.base_url = base_url.rstrip("/")
        self.iterations = iterations
        self.concurrent_users = concurrent_users
        self.output_path = output_path
        self.with_explanation = with_explanation
        self.predict_url = f"{self.base_url}/predict"
        if with_explanation:
            self.predict_url += "?include_explanation=true&top_n=3"
        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "base_url": base_url,
                "iterations": iterations,
                "concurrent_users": concurrent_users,
                "with_explanation": with_explanation,
            },
            "health_check": {},
            "cold_start": {},
            "single_request": {},
            "concurrent_requests": {},
        }

    def sample_payload(self) -> Dict:
        """Generate sample transaction payload."""
        return {
            "user_id": 12345,
            "account_age_days": 180,
            "total_transactions_user": 25,
            "avg_amount_user": 250.50,
            "amount": 850.75,
            "country": "US",
            "bin_country": "US",
            "channel": "web",
            "merchant_category": "retail",
            "promo_used": 0,
            "avs_match": 1,
            "cvv_result": 1,
            "three_ds_flag": 1,
            "shipping_distance_km": 12.5,
            "transaction_time": "2024-01-15 14:30:00",
        }

    def check_health(self) -> bool:
        """Verify API is healthy before benchmarking."""
        print("🏥 Checking API health...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            health_data = response.json()

            self.results["health_check"] = {
                "status": health_data.get("status"),
                "model_loaded": health_data.get("model_loaded"),
                "model_version": health_data.get("model_version"),
            }

            if health_data.get("status") == "healthy" and health_data.get("model_loaded"):
                print(f"✓ API is healthy (model v{health_data.get('model_version')})")
                return True
            else:
                print("✗ API is not healthy")
                return False

        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False

    def measure_cold_start(self) -> None:
        """Measure cold start latency (first request after startup)."""
        print("\n❄️  Measuring cold start latency...")
        payload = self.sample_payload()

        try:
            start = time.time()
            response = requests.post(
                self.predict_url, json=payload, timeout=30
            )
            e2e_time = (time.time() - start) * 1000  # ms

            if response.status_code == 200:
                data = response.json()
                server_time = data.get("processing_time_ms", 0)

                self.results["cold_start"] = {
                    "server_processing_ms": round(server_time, 2),
                    "end_to_end_ms": round(e2e_time, 2),
                    "network_overhead_ms": round(e2e_time - server_time, 2),
                }

                print(f"✓ Cold start - Server: {server_time:.2f}ms, E2E: {e2e_time:.2f}ms")
            else:
                print(f"✗ Cold start failed: HTTP {response.status_code}")

        except Exception as e:
            print(f"✗ Cold start measurement failed: {e}")

    def benchmark_single_requests(self) -> None:
        """Benchmark single sequential requests."""
        mode = "with SHAP explanation" if self.with_explanation else "prediction only"
        print(f"\n🔥 Benchmarking {self.iterations} single requests ({mode})...")
        payload = self.sample_payload()

        server_times = []
        e2e_times = []
        network_times = []
        successful_requests = 0

        for i in range(self.iterations):
            try:
                start = time.time()
                response = requests.post(
                    self.predict_url, json=payload, timeout=10
                )
                e2e_time = (time.time() - start) * 1000  # ms

                if response.status_code == 200:
                    data = response.json()
                    server_time = data.get("processing_time_ms", 0)

                    server_times.append(server_time)
                    e2e_times.append(e2e_time)
                    network_times.append(e2e_time - server_time)
                    successful_requests += 1

                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{self.iterations} requests")

            except Exception as e:
                print(f"  ✗ Request {i + 1} failed: {e}")

        if server_times:
            self.results["single_request"] = {
                "total_requests": self.iterations,
                "successful_requests": successful_requests,
                "success_rate": round(successful_requests / self.iterations * 100, 2),
                "server_processing": {
                    "mean_ms": round(statistics.mean(server_times), 2),
                    "median_ms": round(statistics.median(server_times), 2),
                    "p95_ms": round(self._percentile(server_times, 95), 2),
                    "p99_ms": round(self._percentile(server_times, 99), 2),
                    "min_ms": round(min(server_times), 2),
                    "max_ms": round(max(server_times), 2),
                },
                "end_to_end": {
                    "mean_ms": round(statistics.mean(e2e_times), 2),
                    "median_ms": round(statistics.median(e2e_times), 2),
                    "p95_ms": round(self._percentile(e2e_times, 95), 2),
                    "p99_ms": round(self._percentile(e2e_times, 99), 2),
                    "min_ms": round(min(e2e_times), 2),
                    "max_ms": round(max(e2e_times), 2),
                },
                "network_overhead": {
                    "mean_ms": round(statistics.mean(network_times), 2),
                    "median_ms": round(statistics.median(network_times), 2),
                },
            }

            print(f"✓ Single request benchmark complete ({successful_requests}/{self.iterations} successful)")

    def benchmark_concurrent_requests(self) -> None:
        """Benchmark concurrent requests with thread pool."""
        mode = "with SHAP explanation" if self.with_explanation else "prediction only"
        print(f"\n⚡ Benchmarking {self.concurrent_users} concurrent users ({mode})...")
        payload = self.sample_payload()
        predict_url = self.predict_url  # Capture for closure

        def make_request():
            """Make a single request and return timing."""
            try:
                start = time.time()
                response = requests.post(
                    predict_url, json=payload, timeout=10
                )
                e2e_time = (time.time() - start) * 1000  # ms

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "server_time": data.get("processing_time_ms", 0),
                        "e2e_time": e2e_time,
                    }
                else:
                    return {"success": False}

            except Exception:
                return {"success": False}

        # Run concurrent requests
        server_times = []
        e2e_times = []
        successful_requests = 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
            futures = [executor.submit(make_request) for _ in range(self.iterations)]

            for future in as_completed(futures):
                result = future.result()
                if result.get("success"):
                    server_times.append(result["server_time"])
                    e2e_times.append(result["e2e_time"])
                    successful_requests += 1

        total_time = time.time() - start_time

        if server_times:
            throughput = successful_requests / total_time

            self.results["concurrent_requests"] = {
                "concurrent_users": self.concurrent_users,
                "total_requests": self.iterations,
                "successful_requests": successful_requests,
                "success_rate": round(successful_requests / self.iterations * 100, 2),
                "total_time_seconds": round(total_time, 2),
                "throughput_rps": round(throughput, 2),
                "server_processing": {
                    "mean_ms": round(statistics.mean(server_times), 2),
                    "median_ms": round(statistics.median(server_times), 2),
                    "p95_ms": round(self._percentile(server_times, 95), 2),
                    "p99_ms": round(self._percentile(server_times, 99), 2),
                },
                "end_to_end": {
                    "mean_ms": round(statistics.mean(e2e_times), 2),
                    "median_ms": round(statistics.median(e2e_times), 2),
                    "p95_ms": round(self._percentile(e2e_times, 95), 2),
                    "p99_ms": round(self._percentile(e2e_times, 99), 2),
                },
            }

            print(f"✓ Concurrent benchmark complete ({successful_requests}/{self.iterations} successful)")
            print(f"  Throughput: {throughput:.2f} requests/second")

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def print_report(self) -> None:
        """Print formatted benchmark report."""
        print("\n" + "=" * 80)
        print("FRAUD DETECTION API - PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        # Metadata
        print(f"\n📊 Test Configuration:")
        print(f"  Timestamp: {self.results['metadata']['timestamp']}")
        print(f"  Base URL: {self.results['metadata']['base_url']}")
        print(f"  Iterations: {self.results['metadata']['iterations']}")
        print(f"  Concurrent Users: {self.results['metadata']['concurrent_users']}")
        mode = "with SHAP explanation" if self.results['metadata'].get('with_explanation') else "prediction only"
        print(f"  Mode: {mode}")

        # Health Check
        if self.results.get("health_check"):
            health = self.results["health_check"]
            print(f"\n🏥 Health Check:")
            print(f"  Status: {health.get('status')}")
            print(f"  Model Loaded: {health.get('model_loaded')}")
            print(f"  Model Version: {health.get('model_version')}")

        # Cold Start
        if self.results.get("cold_start"):
            cold = self.results["cold_start"]
            print(f"\n❄️  Cold Start Performance:")
            print(f"  Server Processing: {cold.get('server_processing_ms')} ms")
            print(f"  End-to-End: {cold.get('end_to_end_ms')} ms")
            print(f"  Network Overhead: {cold.get('network_overhead_ms')} ms")

        # Single Request
        if self.results.get("single_request"):
            single = self.results["single_request"]
            print(f"\n🔥 Single Request Performance ({single.get('successful_requests')}/{single.get('total_requests')} successful):")

            print(f"  Server Processing Time:")
            print(f"    Mean:   {single['server_processing']['mean_ms']} ms")
            print(f"    Median: {single['server_processing']['median_ms']} ms")
            print(f"    P95:    {single['server_processing']['p95_ms']} ms")
            print(f"    P99:    {single['server_processing']['p99_ms']} ms")
            print(f"    Range:  {single['server_processing']['min_ms']} - {single['server_processing']['max_ms']} ms")

            print(f"  End-to-End Latency:")
            print(f"    Mean:   {single['end_to_end']['mean_ms']} ms")
            print(f"    Median: {single['end_to_end']['median_ms']} ms")
            print(f"    P95:    {single['end_to_end']['p95_ms']} ms")
            print(f"    P99:    {single['end_to_end']['p99_ms']} ms")

            print(f"  Network Overhead:")
            print(f"    Mean:   {single['network_overhead']['mean_ms']} ms")
            print(f"    Median: {single['network_overhead']['median_ms']} ms")

        # Concurrent Request
        if self.results.get("concurrent_requests"):
            concurrent = self.results["concurrent_requests"]
            print(f"\n⚡ Concurrent Request Performance ({concurrent.get('concurrent_users')} users):")
            print(f"  Throughput: {concurrent.get('throughput_rps')} requests/second")
            print(f"  Success Rate: {concurrent.get('success_rate')}%")
            print(f"  Total Time: {concurrent.get('total_time_seconds')} seconds")

            print(f"  Server Processing Time:")
            print(f"    Mean:   {concurrent['server_processing']['mean_ms']} ms")
            print(f"    Median: {concurrent['server_processing']['median_ms']} ms")
            print(f"    P95:    {concurrent['server_processing']['p95_ms']} ms")
            print(f"    P99:    {concurrent['server_processing']['p99_ms']} ms")

            print(f"  End-to-End Latency:")
            print(f"    Mean:   {concurrent['end_to_end']['mean_ms']} ms")
            print(f"    Median: {concurrent['end_to_end']['median_ms']} ms")
            print(f"    P95:    {concurrent['end_to_end']['p95_ms']} ms")
            print(f"    P99:    {concurrent['end_to_end']['p99_ms']} ms")

        print("\n" + "=" * 80)

    def save_report(self) -> None:
        """Save benchmark results to JSON file."""
        output_file = Path(self.output_path)
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {output_file.absolute()}")

    def run(self) -> None:
        """Run complete benchmark suite."""
        print("\n" + "=" * 80)
        print("FRAUD DETECTION API - PERFORMANCE BENCHMARK")
        print("=" * 80)

        # Check health
        if not self.check_health():
            print("\n❌ Benchmark aborted: API is not healthy")
            return

        # Run benchmarks
        self.measure_cold_start()
        self.benchmark_single_requests()
        self.benchmark_concurrent_requests()

        # Print and save report
        self.print_report()
        self.save_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark Fraud Detection API performance"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of requests per test (default: 100)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--with-explanation",
        action="store_true",
        help="Include SHAP explanation in requests (slower, tests explainability feature)",
    )
    # Default output to benchmarks/results/ directory
    script_dir = Path(__file__).parent
    default_output = script_dir / "results" / "benchmark_results.json"

    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output),
        help=f"Output file path (default: {default_output})",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    benchmark = FraudAPIBenchmark(
        base_url=args.url,
        iterations=args.iterations,
        concurrent_users=args.concurrent,
        output_path=args.output,
        with_explanation=args.with_explanation,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
