#pragma once

#include <string>
#include <topstx/topstx.h>

namespace deep_ep {

// The domain is created on first use and destroyed when the process exits.
class DeepEpTracer {
public:
    static DeepEpTracer& GetInstance() {
        static DeepEpTracer inst;
        return inst;
    }
    ~DeepEpTracer() { topstxDomainDestroy(domain_); }
    topstxDomainHandle_t domain_;

private:
    DeepEpTracer() { domain_ = topstxDomainCreate("deep_ep"); }
    DeepEpTracer(const DeepEpTracer&) = delete;
    DeepEpTracer& operator=(const DeepEpTracer&) = delete;
};

class DeepEpTracepoint {
public:
    explicit DeepEpTracepoint(topstxDomainHandle_t domain)
        : domain_(domain), active_(false) {}
    ~DeepEpTracepoint() { pop(); }

    void push(const char* name) {
        if (!active_) {
            topstxEventAttributes_t event = {};
            event.version = TOPSTX_VERSION;
            event.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType = TOPSTX_MESSAGE_TYPE_STRING;
            event.message.str = name;
            topstxDomainRangePush(domain_, &event);
            active_ = true;
        }
    }

    // Overload with payload: carries extra info (e.g. JSON string)
    void push(const char* name, const char* info) {
        if (!active_) {
            payload_ = info != nullptr ? info : "";
            topstxEventAttributes_t event = {};
            event.version = TOPSTX_VERSION;
            event.size = TOPSTX_EVENT_ATTRIB_STRUCT_SIZE;
            event.messageType = TOPSTX_MESSAGE_TYPE_STRING;
            event.message.str = name;
            event.payloadType = TOPSTX_PAYLOAD_TYPE_STRING;
            event.payload.stringValue = payload_.c_str();
            topstxDomainRangePush(domain_, &event);
            active_ = true;
        }
    }

    void push(const char* name, const std::string& info) {
        push(name, info.c_str());
    }

    void pop() {
        if (active_) {
            topstxDomainRangePop(domain_);
            active_ = false;
        }
    }

private:
    DeepEpTracepoint(const DeepEpTracepoint&) = delete;
    DeepEpTracepoint& operator=(const DeepEpTracepoint&) = delete;

    topstxDomainHandle_t domain_;
    bool active_;
    std::string payload_;
};


// Build a JSON object payload for DEEP_EP_TRACE.
// Example:
//   TraceJson().add("tokens", 1024).add("hidden", 7168).addRaw("matrix", "[[1,2]]")
class TraceJson {
public:
    TraceJson& add(const char* key, int val) {
        append_key(key);
        str_ += std::to_string(val);
        return *this;
    }

    TraceJson& addRaw(const char* key, const std::string& json_value) {
        append_key(key);
        str_ += json_value;
        return *this;
    }

    TraceJson& finish() {
        if (!closed_) {
            str_ += '}';
            closed_ = true;
        }
        return *this;
    }

    const std::string& str() const { return str_; }
    const char* c_str() const {
        if (!closed_) {
            const_cast<TraceJson*>(this)->finish();
        }
        return str_.c_str();
    }

private:
    void append_key(const char* key) {
        if (!first_) {
            str_ += ',';
        }
        first_ = false;
        str_ += '"';
        str_ += key;
        str_ += "\":";
    }

    bool first_ = true;
    bool closed_ = false;
    std::string str_ = "{";
};

// Example:
//   auto info = TraceInfo().add("tokens", 1024).add("hidden", 7168).add("experts", 256);
//   DEEP_EP_TRACE(intranode_dispatch, info);
class TraceInfo {
public:
    TraceInfo& add(const char* key, int val) {
        if (!str_.empty()) str_ += ',';
        str_ += key;
        str_ += '=';
        str_ += std::to_string(val);
        return *this;
    }
    TraceInfo& add(const char* key, const char* val) {
        if (!str_.empty()) str_ += ',';
        str_ += key;
        str_ += '=';
        str_ += val;
        return *this;
    }
    const char* c_str() const { return str_.c_str(); }

private:
    std::string str_;
};

inline const char* trace_payload(const char* value) {
    return value != nullptr ? value : "";
}

inline const char* trace_payload(const std::string& value) {
    return value.c_str();
}

inline const char* trace_payload(const TraceInfo& value) {
    return value.c_str();
}

inline const char* trace_payload(const TraceJson& value) {
    return value.c_str();
}

} // namespace deep_ep

// Declare a scoped tracepoint with domain="deep_ep", name=api_name,
// payload from a TraceInfo (or any object with a .c_str() method / raw const char*).
// The range ends automatically when the enclosing scope exits (RAII).
//
// Usage:
//   DEEP_EP_TRACE(intranode_dispatch,
//                 deep_ep::TraceInfo().add("tokens", num_tokens).add("hidden", hidden));
#define DEEP_EP_TRACE(api_name, info)                                              \
    do {                                                                           \
        deep_ep::DeepEpTracepoint _deep_ep_tp_##api_name(                           \
            deep_ep::DeepEpTracer::GetInstance().domain_);                         \
        _deep_ep_tp_##api_name.push(#api_name, deep_ep::trace_payload(info));      \
    } while (false)
