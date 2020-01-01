extern "C" {
    
#ifndef DTYPE
#define DTYPE float
#endif

    __global__ void vector_equals (const int n,
                                   const DTYPE* x, const int offset_x, const int stride_x,
                                   const DTYPE* y, const int offset_y, const int stride_y,
                                   int* eq_flag) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int ix = offset_x + gid * stride_x;
            const int iy = offset_y + gid * stride_y;
            if (x[ix] != y[iy]) {
                eq_flag[0]++;
            }
        }
    }
}
