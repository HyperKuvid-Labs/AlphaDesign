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
