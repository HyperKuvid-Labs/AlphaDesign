#!/bin/bash

echo "Testing queue_server.cpp compilation and basic functionality..."

cat > test_queue.cpp << 'TESTCODE'
#include "queue_server.cpp"
#include <iostream>

int main() {
    QueueServer<int, 1024> queue;

    std::cout << "Testing basic enqueue/dequeue..." << std::endl;

    // Test enqueue
    for (int i = 0; i < 100; ++i) {
        if (!queue.enqueue(i)) {
            std::cout << "Failed to enqueue at " << i << std::endl;
            return 1;
        }
    }

    std::cout << "Enqueued 100 items successfully" << std::endl;
    std::cout << "Queue size: " << queue.size() << std::endl;

    // Test dequeue
    for (int i = 0; i < 100; ++i) {
        auto val = queue.dequeue();
        if (!val.has_value()) {
            std::cout << "Failed to dequeue at " << i << std::endl;
            return 1;
        }
        if (val.value() != i) {
            std::cout << "Wrong value at " << i << ": got " << val.value() << std::endl;
            return 1;
        }
    }

    std::cout << "Dequeued 100 items successfully" << std::endl;
    std::cout << "Queue is empty: " << (queue.empty() ? "yes" : "no") << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
TESTCODE

echo "Compiling test..."
g++ -std=c++20 -O3 test_queue.cpp -o test_queue

if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

echo "Running test..."
./test_queue

rm -f test_queue test_queue.cpp

echo ""
echo "If this works, the issue is in the benchmark code, not your queue."
