// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "flash_train/primitive/hygon/kda/g2_runner.hpp"
#include "flash_train/primitive/hygon/gdn/g2_gdn.hpp"

namespace ftrain {
namespace {

constexpr std::int64_t kHeadDim  = 128;
constexpr std::int64_t kMaxElems = (std::int64_t{1} << 31) - 1;

// The shared value checklist: architecture, head dims, size limits, layouts, alignment and the
// numeric types every call has. type_of(port) == want for each (port, want) pair.
template<typename Problem>
Result checkCommon(const Problem& problem, const Constraints& constraints,
                   std::initializer_list<std::pair<std::size_t, FTrainNumericType>> types) {
    if (!kda_g2::deviceSupported(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 GDN kernels target gfx936 only");
    }
    if (problem.hasMemory(Problem::kAqk) || problem.hasMemory(Problem::kAkk)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "GDN FP32 does not use aqk/akk; pass null reserved ports");
    }
    const GdnShape& shape = problem.getShape();
    if (shape.key_dim != kHeadDim || shape.value_dim != kHeadDim) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 GDN supports K = V = 128 only");
    }
    if (shape.tokens <= 0 || shape.heads <= 0 || shape.seqs <= 0 || shape.heads > kMaxElems / kHeadDim ||
        shape.tokens > kMaxElems / (shape.heads * kHeadDim) || shape.seqs > kMaxElems / (shape.heads * 4)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "GDN FP32 requires positive T/H/N, T*H*128 and N*H*4 < 2^31");
    }
    for (std::size_t i = 0; i < problem.size(); ++i) {
        if (!problem.hasMemory(i)) { continue; }
        if (problem.getIndexType(i) != FTRAIN_INDEX_TYPE_CONTINUOUS) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 GDN requires continuously indexed ports");
        }
        if (reinterpret_cast<std::uintptr_t>(problem.getAddress(i)) % 16 != 0) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 GDN requires 16-byte aligned ports");
        }
    }
    for (const auto& [port, want] : types) {
        if (problem.hasMemory(port) && problem.getNumericType(port) != want) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "G2 GDN port has an unsupported numeric type");
        }
    }
    return Result{};
}

gdn_fp32::Shape makeShape(const GdnShape& shape, const GdnAttributes& attributes, bool has_cu) {
    gdn_fp32::Shape s;
    s.tokens         = shape.tokens;
    s.heads          = shape.heads;
    s.seqs           = shape.seqs;
    s.has_cu_seqlens = has_cu;
    s.l2norm_qk      = attributes.isL2NormQk();
    return s;
}

}  // namespace

// ---------------------------------------------------------------- forward

const char* G2GdnForward::getName() const noexcept { return "Fp32GdnForward"; }

std::unique_ptr<PrimitiveBase> G2GdnForward::clone() const { return std::make_unique<G2GdnForward>(*this); }

Result G2GdnForward::isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const {
    using P                      = GdnForwardProblem;
    const FTrainNumericType bf16 = FTRAIN_NUMERIC_TYPE_BF16, fp32 = FTRAIN_NUMERIC_TYPE_FP32;
    return checkCommon(problem, constraints,
                       {
                           {P::kQ,         bf16                     },
                           {P::kK,         bf16                     },
                           {P::kV,         bf16                     },
                           {P::kG,         fp32                     },
                           {P::kBeta,      fp32                     },
                           {P::kCuSeqlens, FTRAIN_NUMERIC_TYPE_INT32},
                           {P::kO,         bf16                     },
                           {P::kQRstd,     fp32                     },
                           {P::kKRstd,     fp32                     },
                           {P::kAqk,       bf16                     },
                           {P::kAkk,       bf16                     },
                           {P::kQn,        fp32                     },
                           {P::kKn,        fp32                     },
                           {P::kH,         fp32                     },
    });
}

std::uint64_t G2GdnForward::getRequiredWorkspaceBytes() const noexcept {
    return gdn_fp32::forwardWorkspaceBytes(shape_);
}

void G2GdnForward::configure(const GdnForwardProblem& problem) {
    using P                         = GdnForwardProblem;
    const GdnAttributes& attributes = problem.getAttributes();
    shape_                          = makeShape(problem.getShape(), attributes, problem.hasMemory(P::kCuSeqlens));
    shape_.has_qkn                  = problem.hasMemory(P::kQn);
    args_.q                         = problem.getAddress(P::kQ);
    args_.k                         = problem.getAddress(P::kK);
    args_.v                         = problem.getAddress(P::kV);
    args_.g                         = problem.getAddress(P::kG);
    args_.beta                      = problem.getAddress(P::kBeta);
    args_.cu_seqlens                = static_cast<const int*>(problem.getAddress(P::kCuSeqlens));
    args_.o                         = problem.getAddress(P::kO);
    args_.q_rstd                    = problem.getAddress(P::kQRstd);
    args_.k_rstd                    = problem.getAddress(P::kKRstd);
    args_.qn                        = problem.getAddress(P::kQn);
    args_.kn                        = problem.getAddress(P::kKn);
    args_.saved_h                   = problem.getAddress(P::kH);
    args_.scale                     = attributes.getScale();
}

void G2GdnForward::executeImpl(const Resources& resources) {
    gdn_fp32::forward(shape_, args_, resources.getWorkspace(), resources.getStream());
}

// ---------------------------------------------------------------- backward

const char* G2GdnBackward::getName() const noexcept { return "Fp32GdnBackward"; }

std::unique_ptr<PrimitiveBase> G2GdnBackward::clone() const { return std::make_unique<G2GdnBackward>(*this); }

Result G2GdnBackward::isApplicable(const GdnBackwardProblem& problem, const Constraints& constraints) const {
    using P                      = GdnBackwardProblem;
    const FTrainNumericType bf16 = FTRAIN_NUMERIC_TYPE_BF16, fp32 = FTRAIN_NUMERIC_TYPE_FP32;
    const FTrainNumericType gt = fp32;
    return checkCommon(problem, constraints,
                       {
                           {P::kDO,        bf16                     },
                           {P::kQ,         bf16                     },
                           {P::kK,         bf16                     },
                           {P::kV,         bf16                     },
                           {P::kG,         gt                       },
                           {P::kBeta,      fp32                     },
                           {P::kCuSeqlens, FTRAIN_NUMERIC_TYPE_INT32},
                           {P::kQRstd,     fp32                     },
                           {P::kKRstd,     fp32                     },
                           {P::kAqk,       bf16                     },
                           {P::kAkk,       bf16                     },
                           {P::kQn,        fp32                     },
                           {P::kKn,        fp32                     },
                           {P::kDq,        bf16                     },
                           {P::kDk,        bf16                     },
                           {P::kDv,        bf16                     },
                           {P::kDg,        gt                       },
                           {P::kDBeta,     fp32                     },
                           {P::kH,         fp32                     },
    });
}

std::uint64_t G2GdnBackward::getRequiredWorkspaceBytes() const noexcept {
    return gdn_fp32::backwardWorkspaceBytes(shape_);
}

void G2GdnBackward::configure(const GdnBackwardProblem& problem) {
    using P                         = GdnBackwardProblem;
    const GdnAttributes& attributes = problem.getAttributes();
    shape_                          = makeShape(problem.getShape(), attributes, problem.hasMemory(P::kCuSeqlens));
    shape_.has_saved_h              = problem.hasMemory(P::kH);
    shape_.has_qkn                  = problem.hasMemory(P::kQn);
    args_.d_o                       = problem.getAddress(P::kDO);
    args_.q                         = problem.getAddress(P::kQ);
    args_.k                         = problem.getAddress(P::kK);
    args_.v                         = problem.getAddress(P::kV);
    args_.g                         = problem.getAddress(P::kG);
    args_.beta                      = problem.getAddress(P::kBeta);
    args_.cu_seqlens                = static_cast<const int*>(problem.getAddress(P::kCuSeqlens));
    args_.q_rstd                    = problem.getAddress(P::kQRstd);
    args_.k_rstd                    = problem.getAddress(P::kKRstd);
    args_.qn                        = problem.getAddress(P::kQn);
    args_.kn                        = problem.getAddress(P::kKn);
    args_.dq                        = problem.getAddress(P::kDq);
    args_.dk                        = problem.getAddress(P::kDk);
    args_.dv                        = problem.getAddress(P::kDv);
    args_.dg                        = problem.getAddress(P::kDg);
    args_.dbeta                     = problem.getAddress(P::kDBeta);
    args_.saved_h                   = problem.getAddress(P::kH);
    args_.scale                     = attributes.getScale();
}

void G2GdnBackward::executeImpl(const Resources& resources) {
    gdn_fp32::backward(shape_, args_, resources.getWorkspace(), resources.getStream());
}

}  // namespace ftrain
