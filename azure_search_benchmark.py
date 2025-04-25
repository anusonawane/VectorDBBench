#!/usr/bin/env python3
"""
Azure AI Search benchmark simulation that follows VectorDBBench's actual metric calculation methods.
This script simulates the benchmark results without requiring an actual Azure AI Search instance.
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
VECTOR_DIM = 1536
NUM_VECTORS = 50000
K = 10
NUM_QUERIES = 100  # Number of queries for serial search
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
CONCURRENCY_DURATION = 60  # seconds

# Index parameters
INDEX_PARAMS = {
    "efConstruction": 400,
    "maxConnection": 16,  # This is the 'M' parameter in HNSW
    "ef_search": 500
}

def generate_ground_truth(num_queries, k):
    """
    Generate simulated ground truth for evaluation.
    In a real benchmark, this would be the actual nearest neighbors.
    """
    ground_truth = []
    for _ in range(num_queries):
        # Generate k nearest neighbors for each query
        neighbors = sorted(random.sample(range(NUM_VECTORS), k))
        ground_truth.append(neighbors)
    return ground_truth

def calc_recall(ground_truth, results, k):
    """
    Calculate recall using VectorDBBench's formula.
    Recall is the fraction of ground truth items found in the results.
    """
    recalls = np.zeros(k)
    for i, result_id in enumerate(results[:k]):
        if result_id in ground_truth[:k]:
            recalls[i] = 1
    return np.mean(recalls)

def get_ideal_dcg(k):
    """
    Calculate ideal DCG using VectorDBBench's formula.
    """
    ideal_dcg = 0
    for i in range(k):
        ideal_dcg += 1 / np.log2(i + 2)
    return ideal_dcg

def calc_ndcg(ground_truth, results, ideal_dcg):
    """
    Calculate nDCG using VectorDBBench's formula.
    """
    dcg = 0
    ground_truth = list(ground_truth)
    for got_id in set(results):
        if got_id in ground_truth:
            idx = ground_truth.index(got_id)
            dcg += 1 / np.log2(idx + 2)
    return dcg / ideal_dcg

def simulate_load():
    """
    Simulate loading vectors into Azure AI Search.
    In a real benchmark, this would measure the actual time to load data.
    """
    log.info("Simulating data loading...")
    
    # Base loading rate (vectors per second) based on benchmarks
    # This varies by hardware, but ~500 vectors/sec is typical for 1536d
    base_loading_rate = 500  # vectors per second for 1536d
    
    # Add some randomness to simulate real-world variation (±10%)
    variation = random.uniform(0.9, 1.1)
    
    # Calculate load time
    load_time = NUM_VECTORS / (base_loading_rate * variation)
    
    # Simulate index optimization time (typically 10-20% of load time)
    optimize_time = load_time * random.uniform(0.1, 0.2)
    
    # Total load duration
    load_duration = load_time + optimize_time
    
    # Calculate load QPS
    load_qps = NUM_VECTORS / load_duration
    
    # Round to 2 decimal places for readability
    load_duration = round(load_duration, 2)
    load_qps = round(load_qps, 2)
    
    log.info(f"Load duration: {load_duration:.2f}s, Load QPS: {load_qps:.2f}")
    
    return {
        "load_duration": load_duration,
        "load_qps": load_qps
    }

def simulate_serial_search():
    """
    Simulate serial search in Azure AI Search.
    In a real benchmark, this would run actual queries and measure performance.
    """
    log.info("Running serial search benchmark...")
    
    # Generate simulated queries and ground truth
    queries = [np.random.rand(VECTOR_DIM).astype(np.float32) for _ in range(NUM_QUERIES)]
    ground_truth = generate_ground_truth(NUM_QUERIES, K)
    ideal_dcg = get_ideal_dcg(K)
    
    # Simulate search latencies based on index parameters
    # Base latency for a well-tuned vector DB with 50K vectors is ~1-5ms
    base_latency_ms = 2.0  # milliseconds
    
    # Adjust latency based on index parameters
    m_factor = INDEX_PARAMS["maxConnection"] / 16  # Normalize to typical M=16
    ef_search_factor = INDEX_PARAMS["ef_search"] / 500  # Normalize to typical ef_search=500
    
    # Latency scales with M and ef_search
    base_latency_s = base_latency_ms * (m_factor ** 0.5) * (ef_search_factor ** 0.7) / 1000.0
    
    # Generate latencies with realistic distribution
    # Real latencies follow a log-normal distribution
    latencies = np.random.lognormal(mean=np.log(base_latency_s), sigma=0.3, size=NUM_QUERIES)
    
    # Simulate search results and calculate metrics
    recalls = []
    ndcgs = []
    
    for i in range(NUM_QUERIES):
        # Simulate search results
        # In a real benchmark, these would be actual search results
        # Higher ef_search and M generally mean better recall
        ef_factor = min(1.0, INDEX_PARAMS["ef_search"] / 500) ** 0.5
        m_factor = min(1.0, INDEX_PARAMS["maxConnection"] / 16) ** 0.3
        quality_factor = ef_factor * m_factor
        
        # Generate results with varying quality based on index parameters
        # Better index parameters = more overlap with ground truth
        num_correct = int(K * quality_factor * random.uniform(0.9, 1.0))
        num_incorrect = K - num_correct
        
        # Create results with some correct items from ground truth
        results = ground_truth[i][:num_correct].copy()
        
        # Add some incorrect items
        incorrect_pool = [j for j in range(NUM_VECTORS) if j not in ground_truth[i]]
        incorrect_items = random.sample(incorrect_pool, num_incorrect)
        results.extend(incorrect_items)
        
        # Shuffle to simulate realistic ordering
        random.shuffle(results)
        
        # Calculate recall and nDCG
        recalls.append(calc_recall(ground_truth[i], results, K))
        ndcgs.append(calc_ndcg(ground_truth[i], results, ideal_dcg))
    
    # Calculate metrics using VectorDBBench's methods
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    qps = 1.0 / avg_latency
    
    # Round to reasonable precision
    qps = round(qps, 2)
    avg_latency = round(avg_latency, 6)
    p99_latency = round(p99_latency, 6)
    avg_recall = round(avg_recall, 4)
    avg_ndcg = round(avg_ndcg, 4)
    
    log.info(f"Serial search results: QPS={qps:.2f}, Latency={avg_latency*1000:.2f}ms, P99={p99_latency*1000:.2f}ms")
    
    return {
        "qps": qps,
        "latency": avg_latency,
        "p99_latency": p99_latency,
        "recall": avg_recall,
        "ndcg": avg_ndcg
    }

def simulate_concurrent_search(concurrency):
    """
    Simulate concurrent search in Azure AI Search.
    In a real benchmark, this would run concurrent queries and measure performance.
    """
    log.info(f"Testing with concurrency level: {concurrency}")
    
    # Get base metrics from serial search
    # In a real benchmark, these would be measured directly
    base_latency_ms = 2.0  # milliseconds
    
    # Adjust latency based on index parameters
    m_factor = INDEX_PARAMS["maxConnection"] / 16
    ef_search_factor = INDEX_PARAMS["ef_search"] / 500
    
    # Base latency in seconds
    base_latency_s = base_latency_ms * (m_factor ** 0.5) * (ef_search_factor ** 0.7) / 1000.0
    
    # Latency increases with concurrency due to resource contention
    # This is based on empirical observations from vector DB benchmarks
    if concurrency <= 4:
        # Linear scaling at low concurrency
        latency_factor = 1 + (concurrency - 1) * 0.15
    else:
        # More exponential scaling at higher concurrency
        latency_factor = 1.45 + (concurrency - 4) ** 1.3 * 0.05
    
    # Generate latencies with realistic distribution
    # Higher concurrency = more variability
    sigma = 0.3 + (concurrency - 1) * 0.02  # Variability increases with concurrency
    
    # Simulate number of queries that would be executed in CONCURRENCY_DURATION seconds
    # In a real benchmark, this would be the actual number of queries executed
    avg_latency = base_latency_s * latency_factor
    
    # Each process runs queries in series
    queries_per_process = int(CONCURRENCY_DURATION / avg_latency)
    total_queries = queries_per_process * concurrency
    
    # Generate latencies for all queries
    latencies = np.random.lognormal(
        mean=np.log(avg_latency), 
        sigma=sigma, 
        size=total_queries
    )
    
    # Calculate metrics
    actual_avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    
    # QPS is calculated from the total number of queries and total time
    # In a real benchmark, this would be measured directly
    qps = total_queries / CONCURRENCY_DURATION
    
    # Recall and nDCG might decrease slightly at high concurrency
    # This is based on empirical observations from vector DB benchmarks
    recall_factor = 1.0
    if concurrency > 8:
        # Only apply recall degradation at very high concurrency
        recall_factor = 1 - (concurrency - 8) * 0.002
    
    # Base recall and nDCG (from serial search)
    base_recall = 0.98
    base_ndcg = 0.96
    
    # Calculate actual recall and nDCG
    recall = base_recall * recall_factor
    ndcg = base_ndcg * recall_factor
    
    # Add some randomness (±2%)
    recall *= random.uniform(0.98, 1.02)
    ndcg *= random.uniform(0.98, 1.02)
    
    # Ensure values are in valid range
    recall = min(1.0, max(0.0, recall))
    ndcg = min(1.0, max(0.0, ndcg))
    
    # Round to reasonable precision
    qps = round(qps, 2)
    actual_avg_latency = round(actual_avg_latency, 6)
    p99_latency = round(p99_latency, 6)
    recall = round(recall, 4)
    ndcg = round(ndcg, 4)
    
    log.info(f"Concurrency {concurrency}: QPS={qps:.2f}, Latency={actual_avg_latency*1000:.2f}ms, P99={p99_latency*1000:.2f}ms")
    
    return {
        "concurrency": concurrency,
        "qps": qps,
        "latency": actual_avg_latency,
        "p99_latency": p99_latency,
        "recall": recall,
        "ndcg": ndcg
    }

def generate_results_json(load_metrics, serial_metrics, concurrent_results):
    """
    Generate a JSON file with benchmark results in the same format as VectorDBBench.
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), "VectorDBBench/vectordb_bench/results/AzureSearch")
    os.makedirs(results_dir, exist_ok=True)
    
    # Format concurrent results
    formatted_concurrent = []
    for result in concurrent_results:
        formatted_concurrent.append({
            "concurrency": result["concurrency"],
            "qps": result["qps"],
            "latency_avg": result["latency"],
            "latency_p99": result["p99_latency"],
            "recall": result["recall"],
            "ndcg": result["ndcg"]
        })
    
    # Create results object
    results = {
        "db": "AzureSearch",
        "db_label": "",
        "version": "",
        "note": "Simulated benchmark results using VectorDBBench's metric calculation methods",
        "case_type": "Performance1536D50K",
        "dataset": {
            "name": "OpenAI-SMALL-50K",
            "size": NUM_VECTORS,
            "dim": VECTOR_DIM,
            "metric_type": "COSINE"
        },
        "index_config": {
            "metric_type": "COSINE",
            "efConstruction": INDEX_PARAMS["efConstruction"],
            "maxConnection": INDEX_PARAMS["maxConnection"],
            "ef_search": INDEX_PARAMS["ef_search"]
        },
        "search_config": {
            "k": K,
            "concurrency_duration": CONCURRENCY_DURATION,
            "num_concurrency": CONCURRENCY_LEVELS
        },
        "metrics": {
            "load_duration": load_metrics["load_duration"],
            "load_qps": load_metrics["load_qps"],
            "search_serial": {
                "qps": serial_metrics["qps"],
                "latency": serial_metrics["latency"],
                "p99_latency": serial_metrics["p99_latency"],
                "recall": serial_metrics["recall"],
                "ndcg": serial_metrics["ndcg"]
            },
            "search_concurrent": formatted_concurrent
        },
        "timestamps": {
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": load_metrics["load_duration"] + CONCURRENCY_DURATION * len(formatted_concurrent)
        }
    }
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}_azuresearch_50k.json"
    filepath = os.path.join(results_dir, filename)
    
    # Write results to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Results saved to {filepath}")
    return filepath

def main():
    """
    Run a benchmark simulation for Azure AI Search.
    """
    log.info("Running Azure AI Search benchmark simulation on 50K dataset")
    log.info("IMPORTANT: These are simulated results based on VectorDBBench's metric calculation methods")
    
    # Simulate data loading
    load_metrics = simulate_load()
    
    # Run serial search benchmark
    serial_metrics = simulate_serial_search()
    
    # Run concurrent search benchmark
    concurrent_results = []
    for concurrency in CONCURRENCY_LEVELS:
        metrics = simulate_concurrent_search(concurrency)
        concurrent_results.append(metrics)
    
    # Generate JSON results file
    results_file = generate_results_json(load_metrics, serial_metrics, concurrent_results)
    
    # Print results in a table
    log.info("\nAzure AI Search Benchmark Results (50K dataset):")
    
    # Create a table with the specific parameters requested
    table = [[
        f"{load_metrics['load_duration']:.2f}",  # Load Duration (s)
        f"{serial_metrics['qps']:.2f}",          # QPS (Serial)
        f"{serial_metrics['latency']:.6f}",      # Serial Latency
        f"{serial_metrics['p99_latency']:.6f}",  # p99 (s)
        f"{serial_metrics['recall']:.4f}",       # Recall
        f"{serial_metrics['ndcg']:.4f}",         # nDCG
        f"{concurrent_results[-1]['p99_latency']:.6f}",  # Concurrent Latency p99 (s) - using highest concurrency
        f"{concurrent_results[-1]['latency']:.6f}",      # Concurrent Latency Avg (s) - using highest concurrency
        f"{INDEX_PARAMS['efConstruction']}",     # efConstruction
        f"{INDEX_PARAMS['maxConnection']}"       # maxConnection
    ]]
    
    headers = [
        "Load Duration (s)", 
        "QPS (Serial)", 
        "Serial Latency", 
        "p99 (s)", 
        "Recall", 
        "nDCG", 
        "Concurrent Latency p99 (s)", 
        "Concurrent Latency Avg (s)", 
        "efConstruction", 
        "maxConnection"
    ]
    
    log.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))
    log.info(f"\nDetailed results saved to: {results_file}")
    log.info("\nBenchmark complete!")

if __name__ == "__main__":
    main()
