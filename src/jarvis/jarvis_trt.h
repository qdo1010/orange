#pragma once
// jarvis_trt.h — TensorRT direct-runtime engine wrapper, vendored from red
// (red/src/jarvis_hybridnet.h, namespace jarvis_hn_trt) for lime's JARVIS
// 3D pose path. Engines are pinned to a chosen GPU by the caller: do
// cudaSetDevice(gpu_id) before load_engine() and before any enqueue/memcpy
// on that engine's stream (the Cam2D / Hybrid3D classes handle this).
#if defined(__linux__) || defined(_WIN32)

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace jarvis_hn_trt {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // Errors only. TRT 10.6 emits a benign "engine plan across different
        // models of devices" WARNING because the 2D engines are built on one
        // A16 but run on the other (identical) A16 cam GPUs — validated working,
        // so suppress the noise but keep real errors.
        if (severity <= Severity::kERROR) std::fprintf(stderr, "[HN-TRT] %s\n", msg);
    }
};

inline bool cuda_ok(cudaError_t err, const char *where) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "[HN-TRT] CUDA error in %s: %s\n", where, cudaGetErrorString(err));
    return false;
}

struct Binding {
    nvinfer1::Dims dims{};
    size_t bytes = 0;          // element size * volume
    void *d_ptr = nullptr;     // device buffer
    bool is_input = false;
};

struct Engine {
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    std::unordered_map<std::string, Binding> bindings;  // by tensor name
    bool loaded = false;

    ~Engine() { release(); }

    void release() {
        for (auto &kv : bindings) {
            if (kv.second.d_ptr) cudaFree(kv.second.d_ptr);
        }
        bindings.clear();
        if (stream)  { cudaStreamDestroy(stream); stream = nullptr; }
        if (context) { delete context; context = nullptr; }
        if (engine)  { delete engine;  engine = nullptr; }
        if (runtime) { delete runtime; runtime = nullptr; }
        loaded = false;
    }

    Binding *get(const std::string &name) {
        auto it = bindings.find(name);
        return it == bindings.end() ? nullptr : &it->second;
    }
};

inline size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        default:                          return 4;
    }
}

inline size_t volume(const nvinfer1::Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

// One-shot plugin registry init. Required for engines built with NMS,
// reproLayer, or any other op that ships as a TRT plugin — without this
// deserialization fails with "Cannot deserialize plugin since corresponding
// IPluginCreator not found in Plugin Registry". Safe to call multiple times.
inline void ensure_plugins_registered(nvinfer1::ILogger &logger) {
    static bool initialized = false;
    if (initialized) return;
    if (!initLibNvInferPlugins(&logger, "")) {
        std::fprintf(stderr, "[HN-TRT] WARN: initLibNvInferPlugins returned false "
                             "(continuing, but plugin-bearing engines may fail to load)\n");
    }
    initialized = true;
}

// Deserialize an .engine file, create an execution context, allocate device
// memory for every I/O tensor, and bind tensor addresses. On failure logs
// and returns false; partial state is cleaned up.
inline bool load_engine(Engine &eng, const std::string &engine_path,
                        nvinfer1::ILogger &logger) {
    ensure_plugins_registered(logger);
    eng.release();
    namespace fs = std::filesystem;
    if (!fs::exists(engine_path)) {
        std::fprintf(stderr, "[HN-TRT] engine not found: %s\n", engine_path.c_str());
        return false;
    }
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::fprintf(stderr, "[HN-TRT] cannot open engine: %s\n", engine_path.c_str());
        return false;
    }
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(sz);
    if (!f.read(blob.data(), sz)) {
        std::fprintf(stderr, "[HN-TRT] failed to read engine: %s\n", engine_path.c_str());
        return false;
    }
    f.close();

    eng.runtime = nvinfer1::createInferRuntime(logger);
    if (!eng.runtime) {
        std::fprintf(stderr, "[HN-TRT] createInferRuntime failed for %s\n", engine_path.c_str());
        return false;
    }
    eng.engine = eng.runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!eng.engine) {
        std::fprintf(stderr, "[HN-TRT] deserializeCudaEngine failed for %s\n", engine_path.c_str());
        eng.release();
        return false;
    }
    eng.context = eng.engine->createExecutionContext();
    if (!eng.context) {
        std::fprintf(stderr, "[HN-TRT] createExecutionContext failed for %s\n", engine_path.c_str());
        eng.release();
        return false;
    }
    if (!cuda_ok(cudaStreamCreate(&eng.stream), "cudaStreamCreate")) {
        eng.release();
        return false;
    }

    const int n = eng.engine->getNbIOTensors();

    // Pin every dynamic input to its profile's MAX shape so downstream output
    // shapes resolve. Engines compiled with --min/opt/maxShapes pinned at
    // batch=16 give us the desired shape via MAX; for engines whose inputs
    // are fully static, the build-time shape has no -1's and setInputShape
    // is a no-op-equivalent.
    auto has_dynamic = [](const nvinfer1::Dims &d) {
        for (int i = 0; i < d.nbDims; ++i) if (d.d[i] < 0) return true;
        return false;
    };
    for (int i = 0; i < n; ++i) {
        const char *name = eng.engine->getIOTensorName(i);
        if (eng.engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        auto build_dims = eng.engine->getTensorShape(name);
        if (!has_dynamic(build_dims)) continue;
        auto max_dims = eng.engine->getProfileShape(
            name, 0, nvinfer1::OptProfileSelector::kMAX);
        if (!eng.context->setInputShape(name, max_dims)) {
            std::fprintf(stderr,
                "[HN-TRT] setInputShape failed for %s (engine %s)\n",
                name, engine_path.c_str());
            eng.release();
            return false;
        }
    }
    if (!eng.context->allInputShapesSpecified()) {
        std::fprintf(stderr,
            "[HN-TRT] not all input shapes specified after profile pinning (engine %s)\n",
            engine_path.c_str());
        eng.release();
        return false;
    }

    // Resolve final shapes from the context (handles both static and the
    // dynamic-with-fixed-profile case) and allocate device memory.
    for (int i = 0; i < n; ++i) {
        const char *name = eng.engine->getIOTensorName(i);
        Binding b;
        b.is_input = (eng.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        b.dims     = eng.context->getTensorShape(name);
        b.bytes    = dtype_size(eng.engine->getTensorDataType(name)) * volume(b.dims);
        if (b.bytes == 0) {
            std::fprintf(stderr, "[HN-TRT] tensor %s in %s has zero-volume shape\n",
                         name, engine_path.c_str());
            eng.release();
            return false;
        }
        if (!cuda_ok(cudaMalloc(&b.d_ptr, b.bytes), "cudaMalloc binding")) {
            eng.release();
            return false;
        }
        if (!eng.context->setTensorAddress(name, b.d_ptr)) {
            std::fprintf(stderr, "[HN-TRT] setTensorAddress failed for %s\n", name);
            eng.release();
            return false;
        }
        eng.bindings.emplace(std::string(name), b);
    }

    eng.loaded = true;
    return true;
}

// Diagnostic helper: log the engine's I/O shapes (called once at load).
inline void log_engine_io(const Engine &eng, const char *label) {
    std::fprintf(stderr, "[HN-TRT]   %s bindings:\n", label);
    for (const auto &kv : eng.bindings) {
        const auto &d = kv.second.dims;
        char shape[128] = {0};
        int off = 0;
        for (int i = 0; i < d.nbDims && off < (int)sizeof(shape) - 8; ++i) {
            off += std::snprintf(shape + off, sizeof(shape) - off,
                                 i == 0 ? "%d" : "x%d", static_cast<int>(d.d[i]));
        }
        std::fprintf(stderr, "[HN-TRT]     %-18s %-6s [%s]  %zu B\n",
                     kv.first.c_str(), kv.second.is_input ? "input" : "output",
                     shape, kv.second.bytes);
    }
}

// Single-input convenience: H→D, enqueue, D→H, sync.
// `host_in` is the input tensor's host data (bytes match the binding).
// `host_out` is filled from `out_name`'s device buffer.
inline bool run_single_io(Engine &eng,
                          const std::string &in_name, const float *host_in,
                          const std::string &out_name, float *host_out) {
    Binding *bi = eng.get(in_name);
    Binding *bo = eng.get(out_name);
    if (!bi || !bo) {
        std::fprintf(stderr, "[HN-TRT] missing binding: %s or %s\n",
                     in_name.c_str(), out_name.c_str());
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(bi->d_ptr, host_in, bi->bytes,
                                 cudaMemcpyHostToDevice, eng.stream), "H2D"))
        return false;
    if (!eng.context->enqueueV3(eng.stream)) {
        std::fprintf(stderr, "[HN-TRT] enqueueV3 failed\n");
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(host_out, bo->d_ptr, bo->bytes,
                                 cudaMemcpyDeviceToHost, eng.stream), "D2H"))
        return false;
    return cuda_ok(cudaStreamSynchronize(eng.stream), "stream sync");
}

} // namespace jarvis_hn_trt

#endif // __linux__ || _WIN32
