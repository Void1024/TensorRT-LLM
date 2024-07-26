#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace wo = tensorrt_llm::kernels::weight_only;
namespace ext = tensorrt_llm::cutlass_extensions;

void simple_assert(bool flag)
{
    if (!flag)
    {
        throw std::runtime_error("simple assert failed");
    }
}

struct CudaBuffer
{
    void* _data;
    int _size;

    CudaBuffer(int size_in_bytes)
        : _size(size_in_bytes)
    {
        cudaMalloc(&_data, _size);
    }

    template <typename T = void>
    T* data()
    {
        return reinterpret_cast<T*>(_data);
    }

    void copy_to(void* dst)
    {
        cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
    }

    void copy_from(void* src)
    {
        cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
    }

    ~CudaBuffer()
    {
        cudaFree(_data);
    }
};

template <typename T>
float compare(void* _pa, void* _pb, int size, float scale, bool print = true)
{
    auto pa = reinterpret_cast<T*>(_pa);
    auto pb = reinterpret_cast<T*>(_pb);
    float max_diff = 0.f, tot_diff = 0.f;
    float max_val = 0.f;
    int diff_cnt = 0;
    float threshold = 1e-7;
    for (int n = 0; n < size; ++n)
    {
        float va = static_cast<float>(pa[n]);
        float vb = static_cast<float>(pb[n]);
        max_val = std::max(max_val, vb);
        float diff = std::abs(va - vb);
        if (diff > threshold)
        {
            max_diff = std::max(max_diff, diff);
            tot_diff += diff;
            ++diff_cnt;
        }
        // if(n < 64) {
        //     printf("idx %d, ref %f, val %f\n", n, vb, va);
        // }
    }
    float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
        // error will be larger.
        diff_thres *= 3.f;
    }
    else
#endif
    {
        diff_thres *= 1.5f;
    }
    if (print)
    {
        printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres,
            tot_diff / (diff_cnt + 1e-6), diff_cnt, size);
    }
    return max_diff <= diff_thres;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 minv, T2 maxv)
{
    std::mt19937 gen(rand());
    std::uniform_real_distribution<float> dis(static_cast<float>(minv), static_cast<float>(maxv));
    for (auto& v : vec)
    {
        v = static_cast<T1>(dis(gen));
    }
}

template <typename T>
std::vector<ext::CutlassGemmConfig> get_configs(T& runner, int k)
{
    auto configs = runner.getConfigs();
    std::vector<ext::CutlassGemmConfig> rets;
    for (auto config : configs)
    {
        if (config.stages >= 5)
        {
            continue;
        }
        if (config.split_k_style != ext::SplitKStyle::NO_SPLIT_K)
        {
            int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
            if (k_size % 64)
            {
                continue;
            }
        }
        rets.push_back(config);
    }
    return rets;
}

void print_config_info(ext::CutlassGemmConfig const& config)
{
    using ext::CutlassTileConfig;
    std::string name;
    bool match = false;
#define GET_CONFIG_NAME(TileConfig)                                                                                    \
    if (config.tile_config == TileConfig)                                                                              \
    {                                                                                                                  \
        name = std::string(#TileConfig);                                                                               \
        match = true;                                                                                                  \
    }
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x32);
    GET_CONFIG_NAME(CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x32);
#undef GET_CONFIG_NAME
    simple_assert(match);
    int split_k = config.split_k_factor;
    int stage = config.stages;
    printf("config %s, split k %d, stage %d\n", name.c_str(), split_k, stage);
}

ext::CutlassGemmConfig config(ext::CutlassTileConfig tile_config, int split_k, int stage)
{
    return ext::CutlassGemmConfig{
        tile_config, split_k > 1 ? ext::SplitKStyle::SPLIT_K_SERIAL : ext::SplitKStyle::NO_SPLIT_K, split_k, stage};
}

template <wo::KernelType KT>
struct cutlassTypeMapper
{
};

#define CUTLASS_TYPE_MAPPER_REGISTRY(                                                                                  \
    CudaKernelType, KernelInfoStr, CutlassAType, CutlassWType, WElemBits, CutlassQuantOp)                              \
    template <>                                                                                                        \
    struct cutlassTypeMapper<CudaKernelType>                                                                           \
    {                                                                                                                  \
        using AType = CutlassAType;                                                                                    \
        using WType = CutlassWType;                                                                                    \
        static constexpr cutlass::WeightOnlyQuantOp QuantOp = CutlassQuantOp;                                          \
        static constexpr int WSizeInBits = WElemBits;                                                                  \
        static std::string str(int m, int n, int k, int gs)                                                            \
        {                                                                                                              \
            std::stringstream ss;                                                                                      \
            ss << KernelInfoStr << " mnk(" << m << ", " << n << ", " << k << ")";                                      \
            if (gs != 0)                                                                                               \
                ss << ", gs " << gs;                                                                                   \
            return ss.str();                                                                                           \
        }                                                                                                              \
    };
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4Groupwise, "FP16Int4Groupwise", half, cutlass::uint4b_t, 4,
    cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4Groupwise, "BF16Int4Groupwise", __nv_bfloat16, cutlass::uint4b_t,
    4, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8PerChannel, "FP16Int8PerChannel", half, uint8_t, 8,
    cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8PerChannel, "BF16Int8PerChannel", __nv_bfloat16, uint8_t, 8,
    cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4PerChannel, "FP16Int4PerChannel", half, cutlass::uint4b_t, 4,
    cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4PerChannel, "BF16Int4PerChannel", __nv_bfloat16, cutlass::uint4b_t,
    4, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY);

float run_cuda_kernel(wo::Params& params, int warmup, int iter)
{
    int arch = tensorrt_llm::common::getSMVersion();
    simple_assert(wo::is_supported(arch, params.type));
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i)
    {
        wo::kernel_launcher(arch, params, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        wo::kernel_launcher(arch, params, s);
    }
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <wo::KernelType KT, typename Runner, typename Config>
void exec_cutlass_kernel(
    void* scaled_act, Runner& runner, wo::Params& params, Config& config, char* ws, size_t ws_size, cudaStream_t stream)
{
    using AType = typename cutlassTypeMapper<KT>::AType;
    static constexpr cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp;
    void* act = params.act;
    if (params.act_scale)
    {
        tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<AType, AType>(
            reinterpret_cast<AType*>(scaled_act), reinterpret_cast<AType const*>(params.act),
            reinterpret_cast<AType const*>(params.act_scale), params.m, params.k, stream);
        act = scaled_act;
    }
    if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
    {
        runner.gemm(
            act, params.weight, params.scales, params.out, params.m, params.n, params.k, config, ws, ws_size, stream);
    }
    else if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
    {
        runner.gemm(act, params.weight, params.scales, params.zeros, params.bias, params.out, params.m, params.n,
            params.k, params.groupsize, config, ws, ws_size, stream);
    }
}

template <wo::KernelType KT>
float run_cutlass_kernel(wo::Params& params, int warmup, int iter, ext::CutlassGemmConfig config)
{
    int arch = tensorrt_llm::common::getSMVersion();
    simple_assert(KT == params.type);
    simple_assert(wo::is_supported(arch, params.type));
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    CudaBuffer scaled_act(params.m * params.k * sizeof(AType));
    auto runner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<AType, WType,
        cutlassTypeMapper<KT>::QuantOp>>();
    auto& gemm = *runner;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    int ws_bytes = gemm.getWorkspaceSize(params.m, params.n, params.k);
    char* ws_ptr = nullptr;
    if (ws_bytes)
        cudaMalloc(&ws_ptr, ws_bytes);
    for (int i = 0; i < warmup; ++i)
    {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < iter; ++i)
    {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
    }
    if (ws_ptr)
        cudaFree(ws_ptr);
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(s);
    return time / iter;
}

template <wo::KernelType KT>
class Benchmarker
{
public:
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    static constexpr int ASizeInBits = sizeof(AType) * 8;
    static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;

    Benchmarker(int _m, int _n, int _k, int _gs)
        : m(_m)
        , n(_n)
        , k(_k)
        , gs(_gs)
        , d_act(m * k * ASizeInBits / 8)
        , d_act_scale(k * ASizeInBits / 8)
        , d_weight(k * n * WSizeInBits / 8)
        , d_scales(n * (gs == 0 ? 1 : k / gs) * ASizeInBits / 8)
        , d_zeros(n * (gs == 0 ? 1 : k / gs) * ASizeInBits / 8)
        , d_bias(n * ASizeInBits / 8)
        , d_out(m * n * ASizeInBits / 8)
        , h_ref(m * n)
    {
        std::srand(20240123);
        if constexpr (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
        {
            simple_assert(gs == 0);
        }
        else if (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
        {
            simple_assert(gs == 64 || gs == 128);
        }
        printf("Benchmark %s\n", cutlassTypeMapper<KT>::str(m, n, k, gs).c_str());
        random_init_buffer();
        calculate_ref();
    }

    void random_init_buffer()
    {
        std::vector<AType> h_act(m * k), h_act_scale(k);
        std::vector<uint8_t> h_weight(k * n);
        std::vector<AType> h_scales(n * (gs == 0 ? 1 : k / gs)), h_zeros(n * (gs == 0 ? 1 : k / gs)), h_bias(n);
        random_fill(h_act, -1.f, 1.f);
        random_fill(h_act_scale, -1.f, 1.f);
        random_fill(h_scales, -1.f, 1.f);
        random_fill(h_zeros, -1.f, 1.f);
        random_fill(h_bias, -1.f, 1.f);
        for (uint8_t& v : h_weight)
        {
            v = rand() % 256;
        }

        d_act.copy_from(h_act.data());
        d_act_scale.copy_from(h_act_scale.data());
        d_weight.copy_from(h_weight.data());
        d_scales.copy_from(h_scales.data());
        d_zeros.copy_from(h_zeros.data());
        d_bias.copy_from(h_bias.data());
    }

    void calculate_ref()
    {
        for (int m_idx = 0; m_idx < m; m_idx += 4)
        {
            int remain_m = std::min(m - m_idx, 4);
            void* p_act_scale = nullptr;
            void* p_zeros = nullptr;
            void* p_bias = nullptr;
            if (gs != 0)
            {
                p_zeros = d_zeros.data();
                p_bias = d_bias.data();
                p_act_scale = d_act_scale.data();
            }
            wo::Params params(d_act.data<AType>() + m_idx * k, p_act_scale, d_weight.data(), d_scales.data(), p_zeros,
                p_bias, d_out.data<AType>() + m_idx * n, 1.f, remain_m, n, k, gs, KT);
            run_cuda_kernel(params, 0, 1);
        }
        d_out.copy_to(h_ref.data());
    }

    float run(int warmup, int iter, ext::CutlassGemmConfig config)
    {
        void* p_act_scale = nullptr;
        void* p_zeros = nullptr;
        void* p_bias = nullptr;
        if (gs != 0)
        {
            p_zeros = d_zeros.data();
            p_bias = d_bias.data();
            p_act_scale = d_act_scale.data();
        }
        wo::Params params(d_act.data(), p_act_scale, d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(),
            1.f, m, n, k, gs, KT);
        return run_cutlass_kernel<KT>(params, warmup, iter, config);
    }

    float benchmark(int warmup, int iter, ext::CutlassGemmConfig config, bool print = true)
    {
        float latency = run(warmup, iter, config);
        std::vector<AType> h_out(m * n);
        d_out.copy_to(h_out.data());
        float quant_scale = 1.f / (1 << (WSizeInBits - 1));
        bool pass = compare<AType>(h_out.data(), h_ref.data(), m * n, quant_scale, print);
        print_config_info(config);
        if (pass)
            printf("\033[32m[Verify PASS]\033[0m");
        else
            printf("\033[31m[Verify FAIL]\033[0m");
        printf(" lantecy %.6fus\n", latency);
        if (!pass && !print)
        {
            compare<AType>(h_out.data(), h_ref.data(), m * n, quant_scale, true);
        }
        return latency;
    }

    float benchmark(int warmup, int iter)
    {
        auto runner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<AType, WType,
            cutlassTypeMapper<KT>::QuantOp>>();
        auto& gemm = *runner;
        auto configs = get_configs(gemm, k);
        simple_assert(configs.size() > 0);
        float fast_time = 1e8;
        auto best_config = configs[0];
        for (auto config : configs)
        {
            float time = benchmark(warmup, iter, config, false);
            if (time < fast_time)
            {
                fast_time = time;
                best_config = config;
            }
        }
        printf("\n\033[32mSelect best config!\033[0m\n");
        return benchmark(warmup, iter, best_config, true);
    }

    int m;
    int n;
    int k;
    int gs;
    CudaBuffer d_act;
    CudaBuffer d_act_scale;
    CudaBuffer d_weight;
    CudaBuffer d_scales;
    CudaBuffer d_zeros;
    CudaBuffer d_bias;
    CudaBuffer d_out;
    std::vector<AType> h_ref;
};

TEST(WO, WarpK32)
{
    int warmup = 30, iter = 100;
    Benchmarker<wo::KernelType::FP16Int4Groupwise> benchmarker16(16, 16384, 4096, 128);
    benchmarker16.benchmark(warmup, iter, config(ext::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x32, 2, 3));

    Benchmarker<wo::KernelType::FP16Int4Groupwise> benchmarker32(32, 16384, 4096, 128);
    benchmarker32.benchmark(warmup, iter, config(ext::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x32, 1, 2));
}
