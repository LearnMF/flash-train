// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

#include "gdn_bridge.h"

namespace {
void* currentStream() {
    static const py::object* const current_stream =
        new py::object(py::module_::import("torch").attr("cuda").attr("current_stream"));
    return reinterpret_cast<void*>(py::cast<std::uintptr_t>((*current_stream)().attr("cuda_stream")));
}

using flash_train_torch::GdnBridgeAttributes;
using flash_train_torch::GdnBridgeView;

GdnBridgeView gdnView(const std::optional<at::Tensor>& t) {
    GdnBridgeView v{
        nullptr, {0, 0, 0},
         1, flash_train_torch::kGdnFp32
    };
    if (!t.has_value() || !t->defined() || t->numel() == 0) { return v; }
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "gdn: tensors must be contiguous device tensors");
    TORCH_CHECK(t->dim() >= 1 && t->dim() <= 3, "gdn: tensors must be rank 1..3");
    v.memory   = t->data_ptr();
    v.num_dims = static_cast<std::int32_t>(t->dim());
    for (int64_t i = 0; i < t->dim(); ++i) { v.dims[i] = t->size(i); }
    switch (t->scalar_type()) {
        case at::kBFloat16: v.type = flash_train_torch::kGdnBf16; break;
        case at::kFloat: v.type = flash_train_torch::kGdnFp32; break;
        case at::kInt: v.type = flash_train_torch::kGdnInt32; break;
        default: TORCH_CHECK(false, "gdn: unsupported dtype ", t->scalar_type());
    }
    return v;
}

template<typename Call>
void runGdn(Call call, const std::vector<GdnBridgeView>& views, GdnBridgeAttributes attributes, const at::Tensor& like,
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

// o, q_rstd, k_rstd, aqk, akk, qn, kn, h. q_rstd/k_rstd are empty without l2norm_qk; the
// optional outputs are empty unless requested: save_qkn (l2norm_qk only) -> qn, kn; save_h -> h
// (the recurrent state h).
std::vector<at::Tensor> gdnForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                                   const at::Tensor& beta, const std::optional<at::Tensor>& cu_seqlens, double scale,
                                   bool l2norm_qk, bool save_qkn, bool save_h) {
    TORCH_CHECK(q.dim() == 3 && v.dim() == 3, "gdn: q/k/v must be [T, H, D]");
    const int64_t T = q.size(0), H = q.size(1);
    at::Tensor o      = torch::empty_like(v);
    auto fopt         = q.options().dtype(at::kFloat);
    at::Tensor q_rstd = l2norm_qk ? torch::empty({T, H}, fopt) : torch::empty({0}, fopt);
    at::Tensor k_rstd = l2norm_qk ? torch::empty({T, H}, fopt) : torch::empty({0}, fopt);
    at::Tensor aqk    = torch::empty({0}, q.options());
    at::Tensor akk    = torch::empty({0}, q.options());
    const bool qkn    = save_qkn && l2norm_qk;
    at::Tensor qn     = qkn ? torch::empty(q.sizes(), fopt) : torch::empty({0}, fopt);
    at::Tensor kn     = qkn ? torch::empty(k.sizes(), fopt) : torch::empty({0}, fopt);
    const bool has_cu = cu_seqlens.has_value() && cu_seqlens->defined();
    const int64_t N   = has_cu ? cu_seqlens->numel() - 1 : 1;
    at::Tensor h =
        torch::empty({save_h ? static_cast<int64_t>(flash_train_torch::gdnSavedHElements(T, H, N, has_cu)) : 0}, fopt);
    const std::vector<GdnBridgeView> views = {
        gdnView(q),      gdnView(k),      gdnView(v),   gdnView(g),   gdnView(beta), gdnView(cu_seqlens), gdnView(o),
        gdnView(q_rstd), gdnView(k_rstd), gdnView(aqk), gdnView(akk), gdnView(qn),   gdnView(kn),         gdnView(h),
    };
    runGdn(flash_train_torch::callGdnForward, views, GdnBridgeAttributes{static_cast<float>(scale), l2norm_qk ? 1 : 0},
           q, "ftrainGdnForward");
    return {o, q_rstd, k_rstd, aqk, akk, qn, kn, h};
}

std::vector<at::Tensor> gdnBackward(const at::Tensor& d_o, const std::optional<at::Tensor>& q,
                                    const std::optional<at::Tensor>& k, const at::Tensor& v, const at::Tensor& g,
                                    const at::Tensor& beta, const std::optional<at::Tensor>& cu_seqlens,
                                    const at::Tensor& q_rstd, const at::Tensor& k_rstd, const at::Tensor& aqk,
                                    const at::Tensor& akk, const std::optional<at::Tensor>& qn,
                                    const std::optional<at::Tensor>& kn, const std::optional<at::Tensor>& h,
                                    double scale, bool l2norm_qk) {
    const bool has_qk = q.has_value() && q->defined() && q->numel() > 0;
    TORCH_CHECK(has_qk || (qn.has_value() && qn->defined() && qn->numel() > 0), "gdn: backward needs q/k or qn/kn");
    const at::Tensor& qs           = has_qk ? *q : *qn;  // the [T, H, K] shape and options of q / k
    std::vector<int64_t> dqk_shape = {2};
    dqk_shape.insert(dqk_shape.end(), qs.sizes().begin(), qs.sizes().end());
    at::Tensor dqk = torch::empty(dqk_shape, qs.options().dtype(at::kBFloat16));
    at::Tensor dq = dqk[0], dk = dqk[1], dv = torch::empty_like(v);
    at::Tensor dg = torch::empty_like(g), dbeta = torch::empty_like(beta);
    const std::vector<GdnBridgeView> views = {
        gdnView(d_o),    gdnView(q),      gdnView(k),   gdnView(v),     gdnView(g),  gdnView(beta), gdnView(cu_seqlens),
        gdnView(q_rstd), gdnView(k_rstd), gdnView(aqk), gdnView(akk),   gdnView(qn), gdnView(kn),   gdnView(dq),
        gdnView(dk),     gdnView(dv),     gdnView(dg),  gdnView(dbeta), gdnView(h),
    };
    runGdn(flash_train_torch::callGdnBackward, views, GdnBridgeAttributes{static_cast<float>(scale), l2norm_qk ? 1 : 0},
           qs, "ftrainGdnBackward");
    return {dq, dk, dv, dg, dbeta};
}

}  // namespace
void registerGdn(py::module_& m) {
    m.def("gdn_forward", &gdnForward, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
          py::arg("cu_seqlens"), py::arg("scale"), py::arg("l2norm_qk"), py::arg("save_qkn") = false,
          py::arg("save_h") = false);
    m.def("gdn_backward", &gdnBackward, py::arg("d_o"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"),
          py::arg("beta"), py::arg("cu_seqlens"), py::arg("q_rstd"), py::arg("k_rstd"), py::arg("aqk"), py::arg("akk"),
          py::arg("qn"), py::arg("kn"), py::arg("h"), py::arg("scale"), py::arg("l2norm_qk"));
}
