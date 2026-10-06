// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "kda_bridge.h"

namespace flash_train_torch {

// Defined in flash_train_torch_capi.cpp. The bridge keeps this torch-facing
// translation unit free of the library's HIP-flavored headers: the HCU
// torch build pulls DTK's CUDA-compat headers, which clash with the HIP
// runtime headers inside one translation unit. Plain pointer and integer
// types only.
unsigned char callFp32Gemm(const void* a, const std::int64_t* a_dims, const void* b, const std::int64_t* b_dims,
                           const void* c, const std::int64_t* c_dims, void* d, const std::int64_t* d_dims,
                           const void* alpha, const void* beta, void* stream);

}  // namespace flash_train_torch

namespace {

// The HCU torch wheel's C++ stream headers cannot be included together
// with this TU's world (its CUDA-compat and HIP runtime headers clash), so
// ask torch itself for the current stream. The bound callable is cached
// and intentionally never destroyed.
void* currentStream() {
    static const py::object* const current_stream =
        new py::object(py::module_::import("torch").attr("cuda").attr("current_stream"));
    return reinterpret_cast<void*>(py::cast<std::uintptr_t>((*current_stream)().attr("cuda_stream")));
}

void checkMatrix(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a device tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be FP32");
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank two");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

at::Tensor gemm(const at::Tensor& a, const at::Tensor& b, const std::optional<at::Tensor>& c, double alpha,
                double beta) {
    checkMatrix(a, "a");
    checkMatrix(b, "b");
    TORCH_CHECK(a.size(1) == b.size(0), "a's reduction depth must equal b's rows");
    if (c.has_value()) {
        checkMatrix(*c, "c");
        TORCH_CHECK(c->size(0) == a.size(0) && c->size(1) == b.size(1), "c must be shaped m-by-n");
    }

    at::Tensor d = torch::empty({a.size(0), b.size(1)}, a.options());

    const std::int64_t a_dims[2]  = {a.size(0), a.size(1)};
    const std::int64_t b_dims[2]  = {b.size(0), b.size(1)};
    const std::int64_t cd_dims[2] = {a.size(0), b.size(1)};

    float alpha_value = static_cast<float>(alpha);
    float beta_value  = static_cast<float>(beta);

    void* stream = currentStream();
    // FTRAIN_STATUS_SUCCESS is 0; any other code is a failure.
    const unsigned char status = flash_train_torch::callFp32Gemm(
        a.data_ptr(), a_dims, b.data_ptr(), b_dims, c.has_value() ? c->data_ptr() : nullptr, cd_dims, d.data_ptr(),
        cd_dims, &alpha_value, &beta_value, stream);
    TORCH_CHECK(status == 0, "ftrainGemm failed with status ", static_cast<unsigned int>(status));
    return d;
}

// ------------------------------------------------------------------ KDA
using flash_train_torch::KdaBridgeAttributes;
using flash_train_torch::KdaBridgeView;

KdaBridgeView kdaView(const std::optional<at::Tensor>& t) {
    KdaBridgeView v{
        nullptr, {0, 0, 0},
         1, flash_train_torch::kKdaFp32
    };
    if (!t.has_value() || !t->defined() || t->numel() == 0) { return v; }
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "kda: tensors must be contiguous device tensors");
    TORCH_CHECK(t->dim() >= 1 && t->dim() <= 3, "kda: tensors must be rank 1..3");
    v.memory   = t->data_ptr();
    v.num_dims = static_cast<std::int32_t>(t->dim());
    for (int64_t i = 0; i < t->dim(); ++i) { v.dims[i] = t->size(i); }
    switch (t->scalar_type()) {
        case at::kBFloat16: v.type = flash_train_torch::kKdaBf16; break;
        case at::kFloat: v.type = flash_train_torch::kKdaFp32; break;
        case at::kInt: v.type = flash_train_torch::kKdaInt32; break;
        default: TORCH_CHECK(false, "kda: unsupported dtype ", t->scalar_type());
    }
    return v;
}

template<typename Call>
void runKda(Call call, const std::vector<KdaBridgeView>& views, KdaBridgeAttributes attributes, const at::Tensor& like,
            const char* name) {
    std::uint64_t bytes  = 0;
    const char* message  = "";
    void* stream         = currentStream();
    unsigned char status = call(views.data(), attributes, nullptr, 0, &bytes, stream, &message);
    TORCH_CHECK(status == 0, name, " workspace query failed (status ", static_cast<unsigned int>(status),
                "): ", message);
    // The workspace comes from torch's caching allocator on the current stream.
    at::Tensor workspace = torch::empty({static_cast<int64_t>(bytes)}, like.options().dtype(at::kByte));
    status = call(views.data(), attributes, bytes ? workspace.data_ptr() : nullptr, bytes, nullptr, stream, &message);
    TORCH_CHECK(status == 0, name, " failed (status ", static_cast<unsigned int>(status), "): ", message);
}

// o, q_rstd, k_rstd, aqk, akk (q_rstd/k_rstd are empty without l2norm_qk).
std::vector<at::Tensor> kdaForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                                   const at::Tensor& beta, const std::optional<at::Tensor>& a_log,
                                   const std::optional<at::Tensor>& dt_bias,
                                   const std::optional<at::Tensor>& cu_seqlens, double scale, double lower_bound,
                                   bool gate_in_kernel, bool l2norm_qk) {
    TORCH_CHECK(q.dim() == 3 && v.dim() == 3, "kda: q/k/v must be [T, H, D]");
    const int64_t T = q.size(0), H = q.size(1);
    at::Tensor o                           = torch::empty_like(v);
    auto fopt                              = q.options().dtype(at::kFloat);
    at::Tensor q_rstd                      = l2norm_qk ? torch::empty({T, H}, fopt) : torch::empty({0}, fopt);
    at::Tensor k_rstd                      = l2norm_qk ? torch::empty({T, H}, fopt) : torch::empty({0}, fopt);
    at::Tensor aqk                         = torch::empty({T, H, 64}, q.options());
    at::Tensor akk                         = torch::empty({T, H, 64}, q.options());
    const std::vector<KdaBridgeView> views = {kdaView(q),    kdaView(k),      kdaView(v),       kdaView(g),
                                              kdaView(beta), kdaView(a_log),  kdaView(dt_bias), kdaView(cu_seqlens),
                                              kdaView(o),    kdaView(q_rstd), kdaView(k_rstd),  kdaView(aqk),
                                              kdaView(akk)};
    runKda(flash_train_torch::callKdaForward, views,
           KdaBridgeAttributes{static_cast<float>(scale), static_cast<float>(lower_bound), gate_in_kernel ? 1 : 0,
                               l2norm_qk ? 1 : 0},
           q, "ftrainKdaForward");
    return {o, q_rstd, k_rstd, aqk, akk};
}

// dq, dk, dv, dg, dbeta, da_log, ddt_bias (the last two are empty without gate_in_kernel).
std::vector<at::Tensor> kdaBackward(const at::Tensor& d_o, const at::Tensor& q, const at::Tensor& k,
                                    const at::Tensor& v, const at::Tensor& g, const at::Tensor& beta,
                                    const std::optional<at::Tensor>& a_log, const std::optional<at::Tensor>& dt_bias,
                                    const std::optional<at::Tensor>& cu_seqlens, const at::Tensor& q_rstd,
                                    const at::Tensor& k_rstd, const at::Tensor& aqk, const at::Tensor& akk,
                                    double scale, double lower_bound, bool gate_in_kernel, bool l2norm_qk) {
    at::Tensor dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    at::Tensor dg = torch::empty_like(g), dbeta = torch::empty_like(beta);
    auto fopt                              = q.options().dtype(at::kFloat);
    at::Tensor da_log                      = gate_in_kernel ? torch::empty_like(*a_log) : torch::empty({0}, fopt);
    at::Tensor ddt_bias                    = gate_in_kernel ? torch::empty_like(*dt_bias) : torch::empty({0}, fopt);
    const std::vector<KdaBridgeView> views = {kdaView(d_o),        kdaView(q),      kdaView(k),      kdaView(v),
                                              kdaView(g),          kdaView(beta),   kdaView(a_log),  kdaView(dt_bias),
                                              kdaView(cu_seqlens), kdaView(q_rstd), kdaView(k_rstd), kdaView(aqk),
                                              kdaView(akk),        kdaView(dq),     kdaView(dk),     kdaView(dv),
                                              kdaView(dg),         kdaView(dbeta),  kdaView(da_log), kdaView(ddt_bias)};
    runKda(flash_train_torch::callKdaBackward, views,
           KdaBridgeAttributes{static_cast<float>(scale), static_cast<float>(lower_bound), gate_in_kernel ? 1 : 0,
                               l2norm_qk ? 1 : 0},
           q, "ftrainKdaBackward");
    return {dq, dk, dv, dg, dbeta, da_log, ddt_bias};
}

}  // namespace

PYBIND11_MODULE(_flash_train_torch, m) {
    m.doc() = "flash-train PyTorch bindings";
    m.def("gemm", &gemm, "d = alpha * (a @ b) + beta * c on the current stream", py::arg("a"), py::arg("b"),
          py::arg("c") = std::nullopt, py::arg("alpha") = 1.0, py::arg("beta") = 0.0);
    m.def("kda_forward", &kdaForward, "KDA forward on the current stream -> (o, q_rstd, k_rstd, aqk, akk)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("a_log"), py::arg("dt_bias"),
          py::arg("cu_seqlens"), py::arg("scale"), py::arg("lower_bound"), py::arg("gate_in_kernel"),
          py::arg("l2norm_qk"));
    m.def("kda_backward", &kdaBackward,
          "KDA backward on the current stream -> (dq, dk, dv, dg, dbeta, da_log, ddt_bias)", py::arg("d_o"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("a_log"), py::arg("dt_bias"),
          py::arg("cu_seqlens"), py::arg("q_rstd"), py::arg("k_rstd"), py::arg("aqk"), py::arg("akk"), py::arg("scale"),
          py::arg("lower_bound"), py::arg("gate_in_kernel"), py::arg("l2norm_qk"));
}
