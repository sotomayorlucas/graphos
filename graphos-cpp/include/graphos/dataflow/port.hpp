#pragma once
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include "graphos/core/constants.hpp"
#include "graphos/dataflow/channel.hpp"

namespace graphos {

// ── InputPort<T> — consumer end of a typed channel ──
template <typename T>
class InputPort {
    std::string name_;
    std::shared_ptr<SpscChannel<T>> channel_;

public:
    explicit InputPort(std::string name) : name_(std::move(name)) {}

    const std::string& name() const noexcept { return name_; }
    bool connected() const noexcept { return channel_ != nullptr; }

    void set_channel(std::shared_ptr<SpscChannel<T>> ch) noexcept {
        channel_ = std::move(ch);
    }

    SpscChannel<T>* channel() noexcept { return channel_.get(); }

    // Blocking pop — returns nullopt when channel closed+drained
    std::optional<T> get() {
        if (!channel_) [[unlikely]]
            throw std::runtime_error("InputPort '" + name_ + "' not connected");
        return channel_->pop();
    }

    // Non-blocking
    std::optional<T> try_get() {
        if (!channel_) [[unlikely]]
            throw std::runtime_error("InputPort '" + name_ + "' not connected");
        return channel_->try_pop();
    }
};

// ── OutputPort<T> — producer end of a typed channel ──
template <typename T>
class OutputPort {
    std::string name_;
    std::shared_ptr<SpscChannel<T>> channel_;

public:
    explicit OutputPort(std::string name) : name_(std::move(name)) {}

    const std::string& name() const noexcept { return name_; }
    bool connected() const noexcept { return channel_ != nullptr; }

    void set_channel(std::shared_ptr<SpscChannel<T>> ch) noexcept {
        channel_ = std::move(ch);
    }

    SpscChannel<T>* channel() noexcept { return channel_.get(); }

    // Blocking push
    void put(T&& item) {
        if (!channel_) [[unlikely]]
            throw std::runtime_error("OutputPort '" + name_ + "' not connected");
        channel_->push(std::move(item));
    }

    void put(const T& item) {
        if (!channel_) [[unlikely]]
            throw std::runtime_error("OutputPort '" + name_ + "' not connected");
        channel_->push(item);
    }

    // Signal completion — downstream pop() will return nullopt after drain
    void close() noexcept {
        if (channel_) channel_->close();
    }
};

// ── Connect helper: creates a shared channel between output and input ──
template <typename T>
std::shared_ptr<SpscChannel<T>> connect(OutputPort<T>& out, InputPort<T>& in,
                                         size_t capacity = DEFAULT_BATCH_SIZE) {
    auto ch = std::make_shared<SpscChannel<T>>(capacity);
    out.set_channel(ch);
    in.set_channel(ch);
    return ch;
}

} // namespace graphos
