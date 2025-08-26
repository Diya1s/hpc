#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void matmul(int N, double *A, double *B, double *C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[(long)i*N + k] * B[(long)k*N + j];
            }
            C[(long)i*N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) { 
        printf("Usage: %s N num_threads\n", argv[0]); 
        return 1; 
    }
    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    double *A = (double*)malloc((size_t)N*N*sizeof(double));
    double *B = (double*)malloc((size_t)N*N*sizeof(double));
    double *C = (double*)malloc((size_t)N*N*sizeof(double));
    if (!A || !B || !C) { 
        fprintf(stderr, "malloc failed\n"); 
        return 2; 
    }

    for (long i = 0; i < (long)N*N; i++) { 
        A[i] = 1.0; 
        B[i] = 2.0; 
    }

    double t0 = now_sec();
    matmul(N, A, B, C);
    double t1 = now_sec();

    double elapsed = t1 - t0;
    double gflops = (2.0 * N * (double)N * (double)N) / (elapsed * 1e9);

    printf("OpenMP MatMul: N=%d threads=%d elapsed=%.6f s, perf=%.3f GFLOP/s\n", 
           N, num_threads, elapsed, gflops);

    free(A); free(B); free(C);
    return 0;
}
