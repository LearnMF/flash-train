// Runs kda_g2::forward + backward (scalar_gate = GDN) on the CPU stand-in runtime.
// usage: sim_main <dir> <T> <H> <N> <save_qn> <save_h>   (N > 0: <dir>/cu.bin holds N + 1 int32)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include "flash_train/primitive/hygon/kda/g2_runner.hpp"

namespace {
void* alloc(size_t bytes) {
    void* p = aligned_alloc(4096, (bytes + 8191) / 4096 * 4096);
    memset(p, 0x5a, (bytes + 8191) / 4096 * 4096);   // poison: stale reads must not look like zeros
    return p;
}
void* rd(const std::string& f, size_t bytes) {
    void* p = alloc(bytes);
    FILE* fp = fopen(f.c_str(), "rb");
    if (!fp || fread(p, 1, bytes, fp) != bytes) { fprintf(stderr, "read %s failed\n", f.c_str()); exit(2); }
    fclose(fp);
    return p;
}
void wr(const std::string& f, const void* p, size_t bytes) {
    FILE* fp = fopen(f.c_str(), "wb");
    fwrite(p, 1, bytes, fp);
    fclose(fp);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) return 1;
    using namespace ftrain::kda_g2;
    const std::string d = std::string(argv[1]) + "/";
    const size_t T = atol(argv[2]), H = atol(argv[3]), N = atol(argv[4]);
    const bool save_qn = atoi(argv[5]), save_h = atoi(argv[6]);
    const size_t row = T * H * 128 * 2, sc = T * H * 4;
    Shape s;
    s.tokens = T; s.heads = H; s.seqs = N ? N : 1; s.has_cu_seqlens = N > 0; s.scalar_gate = true;
    ForwardArgs f{};
    f.q = rd(d + "q.bin", row); f.k = rd(d + "k.bin", row); f.v = rd(d + "v.bin", row);
    f.g = rd(d + "g.bin", sc); f.beta = rd(d + "beta.bin", sc);
    f.cu_seqlens = N ? static_cast<const int*>(rd(d + "cu.bin", (N + 1) * 4)) : nullptr;
    f.o = alloc(row); f.q_rstd = alloc(sc); f.k_rstd = alloc(sc);
    f.aqk = alloc(T * H * 64 * 2); f.akk = alloc(T * H * 64 * 2);
    char* qk = static_cast<char*>(alloc(2 * row));
    if (save_qn) { f.qn = qk; f.kn = qk + row; }
    const size_t chunks = s.chunks() * H;
    if (save_h) f.saved_h = alloc(chunks * 16384 * 2);
    f.scale = 0.08838834764831845f;   // 128^-0.5
    forward(s, f, alloc(forwardWorkspaceBytes(s)), nullptr);
    wr(d + "o.bin", f.o, row);

    Shape sb = s;
    sb.has_saved_h = save_h;
    BackwardArgs b{};
    b.d_o = rd(d + "do.bin", row);
    b.q = f.q; b.k = f.k; b.v = f.v; b.g = f.g; b.beta = f.beta; b.cu_seqlens = f.cu_seqlens;
    b.q_rstd = f.q_rstd; b.k_rstd = f.k_rstd; b.aqk = f.aqk; b.akk = f.akk;
    b.qn = f.qn; b.kn = f.kn; b.saved_h = f.saved_h;
    char* dqk = static_cast<char*>(alloc(2 * row));
    b.dq = dqk; b.dk = dqk + row; b.dv = alloc(row); b.dg = alloc(sc); b.dbeta = alloc(sc);
    b.scale = f.scale;
    sb.g_in_dqdk = save_qn;
    backward(sb, b, alloc(backwardWorkspaceBytes(sb)), nullptr);
    wr(d + "dq.bin", b.dq, row); wr(d + "dk.bin", b.dk, row); wr(d + "dv.bin", b.dv, row);
    wr(d + "dg.bin", b.dg, sc); wr(d + "dbeta.bin", b.dbeta, sc);
    printf("kernels launched: %ld\n", g2_fake::n_kernels);
    return 0;
}
