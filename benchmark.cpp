#include "queue_server.cpp"
#include <thread>
#include <chrono>
#include <iomanip>

using namespace std::chrono;

struct BenchmarkResult {
    double ops_per_sec;
    double avg_latency_ns;
    size_t total_ops;
    double duration_sec;
};

template<size_t CAPACITY>
BenchmarkResult benchmark_single_thread(size_t operations) {
    QueueServer<uint64_t, CAPACITY> queue;

    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < operations; ++i) {
        queue.enqueue(i);
    }

    for (size_t i = 0; i < operations; ++i) {
        queue.dequeue();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();

    BenchmarkResult result;
    result.total_ops = operations * 2;
    result.duration_sec = duration / 1e9;
    result.ops_per_sec = result.total_ops / result.duration_sec;
    result.avg_latency_ns = duration / static_cast<double>(result.total_ops);

    return result;
}

template<size_t CAPACITY>
BenchmarkResult benchmark_multi_thread(size_t operations, size_t num_producers, size_t num_consumers) {
    QueueServer<uint64_t, CAPACITY> queue;
    std::atomic<bool> start_flag{false};
    std::atomic<size_t> enqueued{0};
    std::atomic<size_t> dequeued{0};

    auto producer_work = [&](size_t ops_per_thread) {
        while (!start_flag.load()) {}
        for (size_t i = 0; i < ops_per_thread; ++i) {
            while (!queue.enqueue(i)) {
                std::this_thread::yield();
            }
            enqueued.fetch_add(1);
        }
    };

    auto consumer_work = [&](size_t ops_per_thread) {
        while (!start_flag.load()) {}
        for (size_t i = 0; i < ops_per_thread; ++i) {
            while (!queue.dequeue()) {
                std::this_thread::yield();
            }
            dequeued.fetch_add(1);
        }
    };

    size_t ops_per_producer = operations / num_producers;
    size_t ops_per_consumer = operations / num_consumers;

    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_producers; ++i) {
        threads.emplace_back(producer_work, ops_per_producer);
    }

    for (size_t i = 0; i < num_consumers; ++i) {
        threads.emplace_back(consumer_work, ops_per_consumer);
    }

    auto start = high_resolution_clock::now();
    start_flag.store(true);

    for (auto& t : threads) {
        t.join();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start).count();

    BenchmarkResult result;
    result.total_ops = enqueued.load() + dequeued.load();
    result.duration_sec = duration / 1e9;
    result.ops_per_sec = result.total_ops / result.duration_sec;
    result.avg_latency_ns = duration / static_cast<double>(result.total_ops);

    return result;
}

void print_result(const string& name, const BenchmarkResult& result) {
    cout << std::left << std::setw(35) << name
         << " | Ops/sec: " << std::setw(15) << std::fixed << std::setprecision(0) << result.ops_per_sec
         << " | Latency: " << std::setw(10) << std::fixed << std::setprecision(2) << result.avg_latency_ns << " ns"
         << " | Duration: " << std::fixed << std::setprecision(3) << result.duration_sec << "s"
         << endl;
}

int main() {
    cout << "\n=== Lock-Free Queue Server Benchmark ===\n" << endl;
    cout << "Hardware Concurrency: " << std::thread::hardware_concurrency() << " threads\n" << endl;

    const size_t SMALL_OPS = 1'000'000;
    const size_t MEDIUM_OPS = 10'000'000;
    const size_t LARGE_OPS = 50'000'000;

    cout << "--- Single Thread Performance ---" << endl;
    print_result("Small Queue (1K) - 1M ops", benchmark_single_thread<1024>(SMALL_OPS));
    print_result("Medium Queue (64K) - 1M ops", benchmark_single_thread<65536>(SMALL_OPS));
    print_result("Large Queue (1M) - 1M ops", benchmark_single_thread<1048576>(SMALL_OPS));

    cout << "\n--- Multi-Thread Performance (2P/2C) ---" << endl;
    print_result("Small Queue (1K) - 10M ops", benchmark_multi_thread<1024>(MEDIUM_OPS, 2, 2));
    print_result("Medium Queue (64K) - 10M ops", benchmark_multi_thread<65536>(MEDIUM_OPS, 2, 2));
    print_result("Large Queue (1M) - 10M ops", benchmark_multi_thread<1048576>(MEDIUM_OPS, 2, 2));

    cout << "\n--- Multi-Thread Performance (4P/4C) ---" << endl;
    print_result("Small Queue (1K) - 10M ops", benchmark_multi_thread<1024>(MEDIUM_OPS, 4, 4));
    print_result("Medium Queue (64K) - 10M ops", benchmark_multi_thread<65536>(MEDIUM_OPS, 4, 4));
    print_result("Large Queue (1M) - 10M ops", benchmark_multi_thread<1048576>(MEDIUM_OPS, 4, 4));

    cout << "\n--- Stress Test (8P/8C) ---" << endl;
    print_result("Medium Queue (64K) - 50M ops", benchmark_multi_thread<65536>(LARGE_OPS, 8, 8));

    cout << "\n--- High Contention Test (16P/16C) ---" << endl;
    print_result("Small Queue (1K) - 10M ops", benchmark_multi_thread<1024>(MEDIUM_OPS, 16, 16));

    cout << "\n=== Benchmark Complete ===\n" << endl;

    return 0;
}
