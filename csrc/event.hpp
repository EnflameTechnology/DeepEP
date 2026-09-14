#include <memory>

#include "kernels/exception.h"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kPrivateUse1);
        event->record(torch_gcu::getCurrentGCUStream());
    }

    explicit EventHandle(const torch_gcu::GCUStream& stream) {
        event = std::make_shared<torch::Event>(torch::kPrivateUse1);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;
    EventHandle& operator=(const EventHandle& other) = default;

    void current_stream_wait() const {
        torch_gcu::getCurrentGCUStream().unwrap().wait(*event);
    }
};

torch::Event create_event(const torch_gcu::GCUStream &s) {
    auto event = torch::Event(torch::kPrivateUse1);
    event.record(s);
    return event;
}

void stream_wait(const torch_gcu::GCUStream& s_0, const torch_gcu::GCUStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const torch_gcu::GCUStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

} // namespace deep_ep
