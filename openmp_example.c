// File: openmp_example.c
#include <stdio.h>
#include <omp.h>

int main() {
    int i;
    int n = 10;
    int a[10];

    // Parallel region with for loop
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        a[i] = i * i;
        printf("Thread %d: a[%d] = %d\n", omp_get_thread_num(), i, a[i]);
    }

    return 0;
}
