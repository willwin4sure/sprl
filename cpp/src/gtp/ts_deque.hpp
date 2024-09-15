#ifndef GTP_TS_DEQUE_HPP
#define GTP_TS_DEQUE_HPP

/**
 * @file ts_deque.hpp
 * 
 * Thread-safe wrapper around standard double-ended queue,
 * e.g. for storing messages waiting to be processed.
 * 
 * Borrowed from https://github.com/willwin4sure/flash.
*/

#include <condition_variable>
#include <deque>
#include <mutex>

namespace gtp {

/**
 * Thread-safe double-ended queue for storing arbitrary data.
 * 
 * All functions contain a `std::scoped_lock` which is an RAII
 * lock on the mutex protecting the underlying `std::deque`.
 * 
 * @note `std::deque` is actually faster than `std::queue`:
 * https://stackoverflow.com/questions/59542801/why-is-deque-faster-than-queue.
*/
template <typename T>
class ts_deque {
public:
    ts_deque() = default;

    // Don't want to be able to copy the mutex.
    ts_deque(const ts_deque<T>&) = delete;
    ts_deque& operator=(const ts_deque<T>&) = delete;

    /**
     * @returns Whether the deque is empty.
    */
    bool empty() const {
        std::scoped_lock lock { m_mutexDeque };
        return m_deque.empty();
    }

    /**
     * @returns The size of the deque.
    */
    size_t size() const {
        std::scoped_lock lock { m_mutexDeque };
        return m_deque.size();
    }

    /**
     * @returns A const reference to the back of the deque.
     * 
     * @note Undefined behavior if the deque is empty.
    */
    const T& back() const {
        std::scoped_lock lock { m_mutexDeque };
        return m_deque.back();
    }
    
    /**
     * @returns A const reference to the front of the deque.
     * 
     * @note Undefined behavior if the deque is empty.
    */
    const T& front() const {
        std::scoped_lock lock { m_mutexDeque };
        return m_deque.front();
    }

    /**
     * Clears the deque.
    */
    void clear() {
        std::scoped_lock lock { m_mutexDeque };
        m_deque.clear();
    }

    /**
     * Moves and pushes an element to the back of the deque, extending it.
     * 
     * @note javidx9's version of the code passed in a const reference to T
     * and then tried to std::move it into the back of the deque.
     * This is wrong since it just makes a copy instead, as it is a reference to a const:
     * https://stackoverflow.com/questions/28595117/why-can-we-use-stdmove-on-a-const-object.
     * 
     * I have edited the code to take in an r-value reference instead,
     * and the move constructor gets called now. This allows you to still pass
     * in both r-values and moved l-values.
    */
    void push_back(T&& value) {
        std::scoped_lock lock { m_mutexDeque };
        m_deque.emplace_back(std::move(value));

        // Notify any waiting threads that the deque is no longer empty.
        std::unique_lock<std::mutex> ul { m_mutexBlocking };
        m_cvBlocking.notify_one();
    }

    /**
     * Moves and pushes an element to the front of the deque, extending it.
    */
    void push_front(T&& value) {
        std::scoped_lock lock { m_mutexDeque };
        m_deque.emplace_front(std::move(value));

        // Notify any waiting threads that the deque is no longer empty.
        std::unique_lock<std::mutex> ul { m_mutexBlocking };
        m_cvBlocking.notify_one();
    }    

    /**
     * Pop the front of the deque, shortening it.
     * 
     * @returns The element that was popped off the front.
     * 
     * @note If the deque is empty, results in undefined behavior.
    */
    T pop_front() {
        std::scoped_lock lock { m_mutexDeque };
        T value = std::move(m_deque.front());
        m_deque.pop_front();
        return value;
    }

    /**
     * Pop the back of the deque, shortening it.
     * 
     * @returns The element that was popped off the back.
     * 
     * @note If the deque is empty, results in undefined behavior.
    */
    T pop_back() {
        std::scoped_lock lock { m_mutexDeque };
        T value = std::move(m_deque.back());
        m_deque.pop_back();
        return value;
    }

    /**
     * Blocks the current thread until the deque is no longer empty.
    */
    void wait() const {
        // While loop to handle spurious wake-ups.
        while (empty()) {
            std::unique_lock<std::mutex> ul { m_mutexBlocking };
            m_cvBlocking.wait(ul);  // Automatically releases the lock and waits.
        }
    }

protected:
    std::deque<T> m_deque;            // Underlying double-ended queue holding data.
    mutable std::mutex m_mutexDeque;  // Lock around the deque.

    mutable std::condition_variable m_cvBlocking;  // Blocking condition variable to wait for non-empty.
    mutable std::mutex m_mutexBlocking;            // Lock around the blocking condition variable.
};

} // namespace gtp

#endif