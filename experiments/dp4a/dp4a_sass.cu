// Compare SASS for the llama.cpp sm_50 dp4a fallback vs alternatives.
#include <cstdint>
__device__ __forceinline__ int dp4a_ggml(int a, int b, int c) {
    const int8_t * a8 = (const int8_t *) &a; const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}
// Variant: explicit sign-extending byte extracts + 16-bit multiplies (maps to one XMAD each).
__device__ __forceinline__ int dp4a_xmad(int a, int b, int c) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        short ai = (short)(int8_t)(a >> (8*i)); short bi = (short)(int8_t)(b >> (8*i));
        c += (int)__mul24(ai, bi);
    }
    return c;
}
extern "C" __global__ void k_ggml(const int * a, const int * b, int * out, int n) {
    int c = 0; for (int i = threadIdx.x; i < n; i += blockDim.x) c = dp4a_ggml(a[i], b[i], c); out[threadIdx.x] = c; }
extern "C" __global__ void k_xmad(const int * a, const int * b, int * out, int n) {
    int c = 0; for (int i = threadIdx.x; i < n; i += blockDim.x) c = dp4a_xmad(a[i], b[i], c); out[threadIdx.x] = c; }
__device__ __forceinline__ int dp4a_vmad(int a, int b, int c) {
    int r = c;
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;\n\t"
        "vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;" : "+r"(r) : "r"(a), "r"(b));
    return r;
}
extern "C" __global__ void k_vmad(const int * a, const int * b, int * out, int n) {
    int c = 0; for (int i = threadIdx.x; i < n; i += blockDim.x) c = dp4a_vmad(a[i], b[i], c); out[threadIdx.x] = c; }
