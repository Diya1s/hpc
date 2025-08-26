#include <stdio.h>
#include <omp.h>
int main() {
    int i;
    int n = 12;
    #pragma omp parallel for schedule(static, 3)
    for (i = 0; i < n; i++) {
        printf("Thread %d processing iteration %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
