#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel num_threads(4)
    {
        int id = omp_get_thread_num();

        printf("Thread %d before barrier\n", id);
        #pragma omp barrier
        if (id == 0) {
            printf("All threads reached the barrier. Thread %d continuing.\n", id);
        }
    }
    return 0;
}
