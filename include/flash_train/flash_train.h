// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FLASH_TRAIN_H_
#define FTRAIN_FLASH_TRAIN_H_

#include <flash_train/common.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Executes one GEMM with the recommended Primitive.
 *
 * Convenience wrapper over the staged API: it binds the six Tensor ports
 * and the Gemm attributes internally, creates the Plan through the
 * library's selection cache, and executes primitive 0 -- the recommended
 * default -- on stream. It is equivalent to building one Gemm Pattern,
 * creating Ops, filling Args, creating a Plan, and executing its first
 * Primitive.
 *
 * All staged-API rules apply unchanged: a, b, c, and d are rank-two device
 * Tensors with consistent shapes (c's memory may be null, which drops the
 * beta * c term); alpha and beta are host scalar Tensors; execution is
 * asynchronous on stream, and the referenced memory and workspace must
 * stay valid until the enqueued work completes. The workspace must be at
 * least the recommended Primitive's requirement -- null and zero are
 * accepted when it requires none.
 *
 * @param a Device m-by-k input Tensor.
 * @param b Device k-by-n input Tensor.
 * @param c Device m-by-n input Tensor; its memory may be null.
 * @param d Device m-by-n output Tensor.
 * @param alpha Host scalar Tensor scaling a @ b.
 * @param beta Host scalar Tensor scaling c.
 * @param compute_type Numeric type used for computation.
 * @param workspace Workspace address.
 * @param workspace_bytes Size of workspace in bytes.
 * @param stream Stream used for execution.
 * @return FTRAIN_STATUS_INVALID_ARGUMENT when a parameter is inconsistent
 * or the workspace is insufficient; FTRAIN_STATUS_UNSUPPORTED when no
 * implementation satisfies the parameters; otherwise, the status of the
 * operation.
 */
FTRAIN_API FTrainStatus ftrainGemm(FTrainStorageView a, FTrainStorageView b, FTrainStorageView c, FTrainStorageView d,
                                   FTrainStorageView alpha, FTrainStorageView beta, FTrainNumericType compute_type,
                                   void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/**
 * @brief The Tensor ports of one KDA forward call (see ftrainPatternAddKdaForward).
 *
 * Packed THD layout: q, k, v, o [T, H, 128] bf16; g [T, H, 128] (bf16 raw gate with
 * gate_in_kernel, else fp32 per-token log decay); beta [T, H] fp32 post-activation; a_log [H]
 * and dt_bias [H, 128] fp32 (gate_in_kernel only); cu_seqlens [N + 1] int32; the saved
 * tensors q_rstd, k_rstd [T, H] fp32 (l2norm_qk only) and aqk, akk [T, H, 64] bf16 are
 * outputs the backward consumes. Unused ports take null memory.
 *
 * state (optional output, bf16 [ftrainKdaStateElements(T, H, N, has_cu_seqlens)], opaque layout):
 * when given, the forward also writes its recurrent state for the backward, which then skips
 * recomputing it. Meant for activation-checkpoint recompute (the forward runs right before the
 * backward, so the buffer is short-lived); results are bit-identical either way.
 */
typedef struct {
    FTrainStorageView q, k, v, g, beta, a_log, dt_bias, cu_seqlens;
    FTrainStorageView o, q_rstd, k_rstd, aqk, akk;
    FTrainStorageView state;
} FTrainKdaForwardTensors;

/** @brief The Tensor ports of one KDA backward call (see ftrainPatternAddKdaBackward); state is
 * the forward's optional state output (null memory when the forward did not write one). The
 * outputs dq, dk, dg also serve as scratch during the call and must not alias any input. */
typedef struct {
    FTrainStorageView d_o, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, q_rstd, k_rstd, aqk, akk;
    FTrainStorageView dq, dk, dv, dg, dbeta, da_log, ddt_bias;
    FTrainStorageView state;
} FTrainKdaBackwardTensors;

/**
 * @brief Element count of the optional KDA state port for T packed tokens, H heads and N
 * sequences: H * C * 32768, with C = T / 64 for one sequence of T % 64 == 0 rows, else
 * ceil(T / 64) + N (the chunk capacity of the packed layout).
 */
static inline uint64_t ftrainKdaStateElements(int64_t tokens, int64_t heads, int64_t seqs, int has_cu_seqlens) {
    const int64_t chunks =
        (!has_cu_seqlens && tokens % 64 == 0) ? tokens / 64 : (tokens + 63) / 64 + (has_cu_seqlens ? seqs : 1);
    return (uint64_t)heads * (uint64_t)chunks * 32768u;
}

/**
 * @brief Returns the workspace bytes the recommended KDA forward Primitive needs.
 *
 * @return FTRAIN_STATUS_INVALID_ARGUMENT for inconsistent ports or attributes,
 * FTRAIN_STATUS_UNSUPPORTED when no implementation applies, otherwise success.
 */
FTRAIN_API FTrainStatus ftrainKdaForwardGetWorkspaceBytes(const FTrainKdaForwardTensors* tensors,
                                                          FTrainKdaAttributes attributes, uint64_t* workspace_bytes);

/**
 * @brief Executes one KDA forward with the recommended Primitive (asynchronous on stream; the
 * memory and workspace must stay valid until the enqueued work completes).
 */
FTRAIN_API FTrainStatus ftrainKdaForward(const FTrainKdaForwardTensors* tensors, FTrainKdaAttributes attributes,
                                         void* workspace, uint64_t workspace_bytes, FTrainStream stream);

/** @brief Returns the workspace bytes the recommended KDA backward Primitive needs. */
FTRAIN_API FTrainStatus ftrainKdaBackwardGetWorkspaceBytes(const FTrainKdaBackwardTensors* tensors,
                                                           FTrainKdaAttributes attributes, uint64_t* workspace_bytes);

/** @brief Executes one KDA backward with the recommended Primitive (asynchronous on stream). */
FTRAIN_API FTrainStatus ftrainKdaBackward(const FTrainKdaBackwardTensors* tensors, FTrainKdaAttributes attributes,
                                          void* workspace, uint64_t workspace_bytes, FTrainStream stream);

#ifdef __cplusplus
}
#endif

#endif
