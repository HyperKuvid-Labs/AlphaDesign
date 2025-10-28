// so this is gonna be the fastest queue server ever, gonna use all my dsa knowledge and cpp skills. Only two operations enqueue and dequeue.

// for now importaing all packages, after use will get a clear picture on what is uses
#include <bits/stdc++.h>

using namespace std;

// json data is gonna be coming in tho, for now let's say strings are incoming
// read this particular repo, and just gonna take what i need
// the repo: https://github.com/andersc/fastqueue

// new things i learnt over here:
// 1. Using atomic operations for thread safety, i thought this is only for cuda, but seems like its a general concept in concurrent programming
// 2. Using cache line alignment to prevent false sharing, seems like a low level optimization but good to know
// 3. [[no discard]] attribute to avoid ignoring return values of functions where it matters, as in for examlple if enqueue fails we should know about it
// 4. no except - used to indicate that a function does not throw exceptions, helps with optimizations and better code generation by the compiler
// 5. is_nothrow_move_assignable_v - trait to check if a type can be moved without throwing exceptions
// 6. memory_order_relaxed, memory_order_acquire, memory_order_release - different memory ordering semantics for atomic operations
// - relaxed: no ordering constraints, only atomicity is guaranteed
// - acquire: ensures that subsequent reads/writes are not moved before this operation
// - release: ensures that previous reads/writes are not moved after this operation
// 7. alignas - used to specify the alignment requirement of a variable or type
// 8. optional - wrapper that may or may not contain a value, useful for functions that might fail or return nothing, like any in ts
template<typename T, size_t CAPACITY>

class QueueServer {
    static_assert(CAPACITY > 0, "queue capacity must be greater than 0");
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "capacity must be power of 2 for performance");

  private:
    static constexpr size_t MASK = CAPACITY-1;
    static constexpr size_t CACHE_LINE = 64;

    // seperae cache lines to prevent false sharing
    alignas(CACHE_LINE) atomic<size_t> head_{0};
    alignas(CACHE_LINE) atomic<size_t> tail_{0};
    alignas(CACHE_LINE) array<T, CAPACITY> buffer_;

    // safety tracking
    alignas(CACHE_LINE) atomic<bool> initialized_{true};

  public:
    QueueServer() noexcept {
        // zero initialixation
        if constexpr (is_trivial_v<T>) {
            buffer_.fill(T{});
        }
    }

    ~QueueServer() noexcept {
        initialized_.store(false, memory_order_release);
    }

    // non-copyable, non-movable for safety
    QueueServer(const QueueServer&) = delete;
    QueueServer& operator=(const QueueServer&) = delete;
    QueueServer(QueueServer&&) = delete;
    QueueServer& operator=(QueueServer&&) = delete;

    [[nodiscard]] bool enqueue(const T& value) noexcept(is_nothrow_copy_assignable_v<T>) {
        if (!initialized_.load(memory_order_acquire)) [[unlikely]] {
            return false;
        }

        const size_t current_head = head_.load(memory_order_relaxed);
        const size_t next_head = (current_head + 1) & MASK;
        const size_t current_tail = tail_.load(memory_order_acquire);

        if (next_head == current_tail) [[unlikely]] {
            return false; // queu is full
        }

        buffer_[current_head] = value;

        head_.store(next_head, memory_order_release);
        return true;
    }

    [[nodiscard]] bool enqueue(T&& value) noexcept(is_nothrow_move_assignable_v<T>) {
        if (!initialized_.load(memory_order_acquire)) [[unlikely]] {
            return false;
        }

        const size_t current_head = head_.load(memory_order_relaxed);
        const size_t next_head = (current_head + 1) & MASK;
        const size_t current_tail = tail_.load(memory_order_acquire);

        if (next_head == current_tail) [[unlikely]] {
            return false;
        }

        buffer_[current_head] = move(value);
        head_.store(next_head, memory_order_release);
        return true;
    }

    [[nodiscard]] optional<T> dequeue() noexcept(is_nothrow_move_constructible_v<T>) {
        if (!initialized_.load(memory_order_acquire)) [[unlikely]] {
            return nullopt;
        }

        const size_t current_tail = tail_.load(memory_order_relaxed);
        const size_t current_head = head_.load(memory_order_acquire);

        if (current_tail == current_head) [[unlikely]] {
            return nullopt; // queue is empty
        }

        T value = move(buffer_[current_tail]);

        const size_t next_tail = (current_tail + 1) & MASK;
        tail_.store(next_tail, memory_order_release);

        return value;
    }

    [[nodiscard]] bool dequeue(T& out) noexcept(is_nothrow_move_assignable_v<T>) {
        if (!initialized_.load(memory_order_acquire)) [[unlikely]] {
            return false;
        }

        const size_t current_tail = tail_.load(memory_order_relaxed);
        const size_t current_head = head_.load(memory_order_acquire);

        if (current_tail == current_head) [[unlikely]] {
            return false;
        }

        out = move(buffer_[current_tail]);
        tail_.store((current_tail + 1) & MASK, memory_order_release);
        return true;
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(memory_order_acquire) ==
               tail_.load(memory_order_acquire);
    }

    [[nodiscard]] bool full() const noexcept {
        const size_t head = head_.load(memory_order_acquire);
        const size_t tail = tail_.load(memory_order_acquire);
        return ((head + 1) & MASK) == tail;
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t head = head_.load(memory_order_acquire);
        const size_t tail = tail_.load(memory_order_acquire);
        return (head - tail) & MASK;
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept {
        return CAPACITY - 1; // One slot reserved for full/empty distinction
    }

    void clear() noexcept {
        tail_.store(head_.load(memory_order_acquire), memory_order_release);
    }

    [[nodiscard]] bool is_valid() const noexcept {
        const size_t head = head_.load(memory_order_acquire);
        const size_t tail = tail_.load(memory_order_acquire);
        return head < CAPACITY && tail < CAPACITY && initialized_.load(memory_order_acquire);
    }
};