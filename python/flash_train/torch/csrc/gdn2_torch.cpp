// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "gdn2_bridge.h"
namespace {
using namespace flash_train_torch;
void run(bool bw, const std::vector<at::Tensor>& tensors, float scale, bool norm) {
    const c10::DeviceGuard guard(tensors[0].device());
    std::vector<Gdn2View> views;
    for (const auto& t : tensors) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == tensors[0].device(),
                    "gdn2: contiguous tensors on one device required");
        TORCH_CHECK(t.dim() >= 1 && t.dim() <= 4, "gdn2: tensor rank must be 1..4");
        TORCH_CHECK(t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kFloat, "gdn2: BF16 or FP32 required");
        Gdn2View v{t.data_ptr(), {}, int(t.dim()), t.scalar_type() == at::kFloat};
        for (int i = 0; i < t.dim(); ++i) v.dims[i] = t.size(i);
        views.push_back(v);
    }
    static const py::object* current_stream =
        new py::object(py::module_::import("torch").attr("cuda").attr("current_stream"));
    void* stream        = reinterpret_cast<void*>(py::cast<std::uintptr_t>((*current_stream)().attr("cuda_stream")));
    std::uint64_t bytes = 0;
    const char* message = "";
    auto status         = callGdn2(bw, views.data(), scale, norm, nullptr, 0, &bytes, stream, &message);
    TORCH_CHECK(!status, "gdn2: ", message);
    auto workspace = at::empty({static_cast<int64_t>(bytes)}, tensors[0].options().dtype(at::kByte));
    status         = callGdn2(bw, views.data(), scale, norm, workspace.data_ptr(), bytes, nullptr, stream, &message);
    TORCH_CHECK(!status, "gdn2: ", message);
}
std::vector<at::Tensor> forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                                const at::Tensor& b, const at::Tensor& w, double scale, bool norm) {
    TORCH_CHECK(q.dim() == 4 && v.dim() == 4, "gdn2: q/v must be BTHD");
    auto n = gdn2SavedElements(q.size(0), q.size(1), q.size(2), v.size(2), q.size(3), v.size(3));
    TORCH_CHECK(n || q.size(1) == 0, "gdn2: unsupported shape");
    auto o     = at::empty(v.sizes(), v.options());
    auto cache = at::empty({static_cast<int64_t>(n)}, q.options().dtype(at::kFloat));
    run(false, {q, k, v, g, b, w, o, cache}, static_cast<float>(scale), norm);
    return {o, cache};
}
std::vector<at::Tensor> backward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& g,
                                 const at::Tensor& b, const at::Tensor& w, const at::Tensor& cache,
                                 const at::Tensor& dout, double scale, bool norm) {
    std::vector<at::Tensor> tensors{q, k, v, g, b, w, cache, dout}, grads;
    for (int i = 0; i < 6; ++i) grads.push_back(at::empty(tensors[i].sizes(), tensors[i].options()));
    tensors.insert(tensors.end(), grads.begin(), grads.end());
    run(true, tensors, static_cast<float>(scale), norm);
    return grads;
}
}  // namespace
void registerGdn2(py::module_& m) {
    m.def("gdn2_forward", &forward);
    m.def("gdn2_backward", &backward);
}
