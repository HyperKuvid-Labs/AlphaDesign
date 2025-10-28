#!/bin/bash

set -e

QUEUE_FILE="queue_server.cpp"
BENCHMARK_CPP="benchmark_prod.cpp"
BINARY="bench_exec"
RESULTS_JSON="benchmark_results_$(date +%Y%m%d_%H%M%S).json"
ITERATIONS=3
WARMUP_RUNS=1

echo "Production-Grade Queue Benchmark"
echo "================================="
echo ""

if [ ! -f "$QUEUE_FILE" ]; then
    echo "Error: $QUEUE_FILE not found"
    exit 1
fi

cat > $BENCHMARK_CPP << 'BENCHCODE'
#include "queue_server.cpp"
#include <thread>
#include <chrono>
#include <vector>
#include <iostream>
#include <atomic>

using namespace std::chrono;

template<size_t CAPACITY>
double benchmark_single(size_t ops) {
    QueueServer<uint64_t, CAPACITY> q;
    auto start = high_resolution_clock::now();

    size_t batch = std::min(ops, CAPACITY / 2);
    for (size_t i = 0; i < batch; ++i) {
        while (!q.enqueue(i)) {}
    }

    for (size_t i = batch; i < ops; ++i) {
        while (!q.enqueue(i)) {}
        while (!q.dequeue()) {}
    }

    while (!q.empty()) {
        while (!q.dequeue()) {}
    }

    auto dur = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
    return (ops * 2) / (dur / 1e9);
}

template<size_t CAPACITY>
double benchmark_multi(size_t ops, size_t prod, size_t cons) {
    QueueServer<uint64_t, CAPACITY> q;
    std::atomic<bool> go{false};
    std::atomic<size_t> done_prod{0};
    std::atomic<size_t> total{0};

    auto producer = [&](size_t n) {
        while (!go.load()) {}
        for (size_t i = 0; i < n; ++i) {
            size_t tries = 0;
            while (!q.enqueue(i)) {
                if (++tries > 1000) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    tries = 0;
                }
            }
            total.fetch_add(1);
        }
        done_prod.fetch_add(1);
    };

    auto consumer = [&](size_t n) {
        while (!go.load()) {}
        for (size_t i = 0; i < n; ++i) {
            while (!q.dequeue()) {
                if (done_prod.load() == prod && q.empty()) break;
                std::this_thread::yield();
            }
            total.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    size_t per_p = ops / prod;
    size_t per_c = ops / cons;

    for (size_t i = 0; i < prod; ++i) threads.emplace_back(producer, per_p);
    for (size_t i = 0; i < cons; ++i) threads.emplace_back(consumer, per_c);

    auto start = high_resolution_clock::now();
    go.store(true);

    for (auto& t : threads) t.join();

    auto dur = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
    return total.load() / (dur / 1e9);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <type> <cap> <ops> <prod> <cons> <warmup>" << std::endl;
        return 1;
    }

    std::string type = argv[1];
    size_t cap = std::stoull(argv[2]);
    size_t ops = std::stoull(argv[3]);
    size_t prod = std::stoull(argv[4]);
    size_t cons = std::stoull(argv[5]);
    bool warmup = std::string(argv[6]) == "1";

    double result = 0;

    try {
        if (type == "single") {
            if (cap == 1024) result = benchmark_single<1024>(ops);
            else if (cap == 65536) result = benchmark_single<65536>(ops);
            else if (cap == 262144) result = benchmark_single<262144>(ops);
        } else {
            if (cap == 1024) result = benchmark_multi<1024>(ops, prod, cons);
            else if (cap == 65536) result = benchmark_multi<65536>(ops, prod, cons);
            else if (cap == 262144) result = benchmark_multi<262144>(ops, prod, cons);
        }
    } catch (...) {
        std::cerr << "Exception occurred" << std::endl;
        return 1;
    }

    if (!warmup && result > 0) {
        std::cout << std::fixed << result << std::endl;
    }

    return 0;
}
BENCHCODE

echo "[1/5] Compiling..."
g++ -std=c++20 -O3 -march=native -pthread -flto -DNDEBUG $BENCHMARK_CPP -o $BINARY 2>&1 | grep -v "warning: ignoring return value" || true

if [ ! -f "$BINARY" ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "[2/5] Running warmup..."
timeout 10 ./$BINARY single 1024 10000 0 0 1 > /dev/null 2>&1 || true

echo "[3/5] Running benchmarks..."
echo ""

declare -A results

run_test() {
    local name=$1
    local type=$2
    local cap=$3
    local ops=$4
    local prod=$5
    local cons=$6

    echo -n "  $name ... "

    local samples=()
    for i in {1..3}; do
        local res=$(timeout 30 ./$BINARY $type $cap $ops $prod $cons 0 2>/dev/null)
        if [ $? -eq 0 ] && [ ! -z "$res" ]; then
            samples+=($res)
        fi
    done

    if [ ${#samples[@]} -eq 0 ]; then
        echo "FAILED"
        echo "0"
    else
        local samples_str=$(IFS=,; echo "${samples[*]}")
        local avg=$(python3 -c "import statistics; print(statistics.mean([$samples_str]))")
        echo "$avg ops/sec"
        echo "$avg"
    fi
}

echo "=== SINGLE-THREADED TESTS ==="
r1=$(run_test "1K/500K" single 1024 500000 0 0)
r2=$(run_test "64K/1M" single 65536 1000000 0 0)
r3=$(run_test "256K/2M" single 262144 2000000 0 0)

echo ""
echo "=== MULTI-THREADED (2P/2C) - LOW CONTENTION ==="
r4=$(run_test "1K/100K" multi 1024 100000 2 2)
r5=$(run_test "64K/500K" multi 65536 500000 2 2)
r6=$(run_test "256K/1M" multi 262144 1000000 2 2)

echo ""
echo "=== MULTI-THREADED (4P/4C) - MEDIUM CONTENTION ==="
r7=$(run_test "1K/100K" multi 1024 100000 4 4)
r8=$(run_test "64K/500K" multi 65536 500000 4 4)
r9=$(run_test "256K/1M" multi 262144 1000000 4 4)

echo ""
echo "=== MULTI-THREADED (8P/8C) - HIGH CONTENTION ==="
r10=$(run_test "1K/100K" multi 1024 100000 8 8)
r11=$(run_test "64K/500K" multi 65536 500000 8 8)
r12=$(run_test "256K/1M" multi 262144 1000000 8 8)

echo ""
echo "=== EXTREME CONTENTION (16P/16C) ==="
r13=$(run_test "1K/50K" multi 1024 50000 16 16)
r14=$(run_test "64K/100K" multi 65536 100000 16 16)

echo ""
echo "=== ASYMMETRIC LOADS ==="
r15=$(run_test "64K/500K (1P/4C)" multi 65536 500000 1 4)
r16=$(run_test "64K/500K (4P/1C)" multi 65536 500000 4 1)
r17=$(run_test "64K/500K (8P/2C)" multi 65536 500000 8 2)

echo ""
echo "=== BURST TESTS (HIGH OPS, SMALL QUEUE) ==="
r18=$(run_test "1K/1M" single 1024 1000000 0 0)
r19=$(run_test "1K/500K (4P/4C)" multi 1024 500000 4 4)

echo ""
echo "=== LATENCY TESTS (LARGE QUEUE, LOW OPS) ==="
r20=$(run_test "256K/100K" single 262144 100000 0 0)
r21=$(run_test "256K/100K (2P/2C)" multi 262144 100000 2 2)

echo ""
echo "[4/5] Calculating statistics..."

CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
CPU_CORES=$(nproc)
MEMORY=$(free -h | awk '/^Mem:/ {print $2}')
CACHE_L1=$(lscpu | grep "L1d cache" | cut -d':' -f2 | xargs)
CACHE_L2=$(lscpu | grep "L2 cache" | cut -d':' -f2 | xargs)
CACHE_L3=$(lscpu | grep "L3 cache" | cut -d':' -f2 | xargs)

echo "[5/5] Creating JSON report..."

cat > $RESULTS_JSON << EOF
{
  "metadata": {
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "iterations": $ITERATIONS,
    "warmup_runs": $WARMUP_RUNS,
    "system": {
      "cpu": "$CPU_MODEL",
      "cores": $CPU_CORES,
      "memory": "$MEMORY",
      "cache": {
        "l1": "$CACHE_L1",
        "l2": "$CACHE_L2",
        "l3": "$CACHE_L3"
      }
    }
  },
  "results": {
    "single_threaded": {
      "1k_500k_ops": $r1,
      "64k_1m_ops": $r2,
      "256k_2m_ops": $r3,
      "1k_1m_ops_burst": $r18,
      "256k_100k_ops_latency": $r20
    },
    "multi_threaded_2p2c": {
      "1k_100k_ops": $r4,
      "64k_500k_ops": $r5,
      "256k_1m_ops": $r6,
      "256k_100k_ops_latency": $r21
    },
    "multi_threaded_4p4c": {
      "1k_100k_ops": $r7,
      "64k_500k_ops": $r8,
      "256k_1m_ops": $r9,
      "1k_500k_ops_burst": $r19
    },
    "multi_threaded_8p8c": {
      "1k_100k_ops": $r10,
      "64k_500k_ops": $r11,
      "256k_1m_ops": $r12
    },
    "extreme_contention_16p16c": {
      "1k_50k_ops": $r13,
      "64k_100k_ops": $r14
    },
    "asymmetric_loads": {
      "64k_500k_1p4c": $r15,
      "64k_500k_4p1c": $r16,
      "64k_500k_8p2c": $r17
    }
  },
  "summary": {
    "best_single_threaded": $(echo "$r1 $r2 $r3" | tr ' ' '\n' | sort -rn | head -1),
    "best_multi_threaded": $(echo "$r4 $r5 $r6 $r7 $r8 $r9 $r10 $r11 $r12" | tr ' ' '\n' | sort -rn | head -1),
    "total_tests": 21
  }
}
EOF

rm -f $BINARY $BENCHMARK_CPP

echo ""
echo "================================================"
echo "           BENCHMARK COMPLETE"
echo "================================================"
echo ""
echo "Results saved to: $RESULTS_JSON"
echo ""
echo "Key Performance Indicators:"
echo "  Best Single-threaded: $(echo "$r1 $r2 $r3" | tr ' ' '\n' | sort -rn | head -1) ops/sec"
echo "  Best Multi-threaded:  $(echo "$r4 $r5 $r6 $r7 $r8 $r9 $r10 $r11 $r12" | tr ' ' '\n' | sort -rn | head -1) ops/sec"
echo ""
echo "Full results:"
cat $RESULTS_JSON | python3 -m json.tool 2>/dev/null || cat $RESULTS_JSON
