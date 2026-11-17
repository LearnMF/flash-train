// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_GDN_H_
#define FTRAIN_GDN_H_

#include "flash_train/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t opaque;
} FTrainGdnForwardOpId;
typedef struct {
    uint64_t opaque;
} FTrainGdnBackwardOpId;
typedef struct {
    float scale;
    int32_t l2norm_qk;
} FTrainGdnAttributes;

/** Packed THD GDN: q/k/v/o bf16 [T,H,128], g/beta fp32 [T,H].
 * g is the per-token log decay, in [-5,0]; beta is post-activation.
 * Scalar decay: S = exp(g)*S; S += beta*k*(v-k^T*S)^T; o = scale*q^T*S.
 * q_rstd/k_rstd fp32 [T,H] are required with l2norm_qk (eps=1e-6).
 * aqk/akk bf16 [T,H,64] are required saved intermediates.
 * cu_seqlens int32 [N+1] is optional; each sequence starts at zero state.
 * qn/kn bf16 [T,H,128] and h bf16 [ftrainGdnSavedHElements(...)] are optional
 * saved tensors. With qn/kn given, backward q/k may be null.
 * The C API uses equal q/k/value heads; the Python API maps grouped heads.
 * Outputs and workspace must not alias inputs or each other. */
typedef struct {
    FTrainStorageView q, k, v, g, beta, cu_seqlens, o, q_rstd, k_rstd, aqk, akk, qn, kn, h;
} FTrainGdnForwardTensors;

FTRAIN_API FTrainStatus ftrainPatternAddGdnForward(FTrainPattern pattern, FTrainGdnForwardOpId* op, FTrainTensorId q,
                                                   FTrainTensorId k, FTrainTensorId v, FTrainTensorId g,
                                                   FTrainTensorId beta, FTrainTensorId cu_seqlens, FTrainTensorId o,
                                                   FTrainTensorId q_rstd, FTrainTensorId k_rstd, FTrainTensorId aqk,
                                                   FTrainTensorId akk, FTrainTensorId qn, FTrainTensorId kn,
                                                   FTrainTensorId h);
FTRAIN_API FTrainStatus ftrainArgsSetGdnForward(FTrainArgs args, FTrainGdnForwardOpId op,
                                                FTrainGdnAttributes attributes);
FTRAIN_API FTrainStatus ftrainGdnForwardGetWorkspaceBytes(const FTrainGdnForwardTensors* tensors,
                                                          FTrainGdnAttributes attributes, uint64_t* workspace_bytes);
FTRAIN_API FTrainStatus ftrainGdnForward(const FTrainGdnForwardTensors* tensors, FTrainGdnAttributes attributes,
                                         void* workspace, uint64_t workspace_bytes, FTrainStream stream);

typedef struct {
    FTrainStorageView d_o, q, k, v, g, beta, cu_seqlens, q_rstd, k_rstd, aqk, akk, qn, kn, dq, dk, dv, dg, dbeta, h;
} FTrainGdnBackwardTensors;

FTRAIN_API FTrainStatus ftrainPatternAddGdnBackward(
    FTrainPattern pattern, FTrainGdnBackwardOpId* op, FTrainTensorId d_o, FTrainTensorId q, FTrainTensorId k,
    FTrainTensorId v, FTrainTensorId g, FTrainTensorId beta, FTrainTensorId cu_seqlens, FTrainTensorId q_rstd,
    FTrainTensorId k_rstd, FTrainTensorId aqk, FTrainTensorId akk, FTrainTensorId qn, FTrainTensorId kn,
    FTrainTensorId dq, FTrainTensorId dk, FTrainTensorId dv, FTrainTensorId dg, FTrainTensorId dbeta, FTrainTensorId h);
FTRAIN_API FTrainStatus ftrainArgsSetGdnBackward(FTrainArgs args, FTrainGdnBackwardOpId op,
                                                 FTrainGdnAttributes attributes);
FTRAIN_API FTrainStatus ftrainGdnBackwardGetWorkspaceBytes(const FTrainGdnBackwardTensors* tensors,
                                                           FTrainGdnAttributes attributes, uint64_t* workspace_bytes);
FTRAIN_API FTrainStatus ftrainGdnBackward(const FTrainGdnBackwardTensors* tensors, FTrainGdnAttributes attributes,
                                          void* workspace, uint64_t workspace_bytes, FTrainStream stream);

static inline uint64_t ftrainGdnSavedHElements(int64_t tokens, int64_t heads, int64_t seqs, int has_cu_seqlens) {
    const int64_t chunks =
        (!has_cu_seqlens && tokens % 64 == 0) ? tokens / 64 : (tokens + 63) / 64 + (has_cu_seqlens ? seqs : 1);
    return (uint64_t)heads * (uint64_t)chunks * 16384u;
}

#ifdef __cplusplus
}
#endif
#endif
