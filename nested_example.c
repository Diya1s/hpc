#include <stdio.h>
#include <omp.h>
int main() {
    omp_set_nested(1); // Enable nested parallelism
    #pragma omp parallel num_threads(2)
    {
        int id = omp_get_thread_num();
        printf("Outer thread %d starting\n", id);

        #pragma omp parallel num_threads(2)
        {
            int inner_id = omp_get_thread_num();
            printf(" Inner thread %d of outer thread %d\n", inner_id, id);
        }
    }
    return 0;
}
