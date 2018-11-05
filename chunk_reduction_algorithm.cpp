#include "gpu_boundary_matrix.h"

class chunk_reduction_algorithm {
public:
    gpu_boundary_matrix gpu_matrix;

public:
    void local_chunk_reduction();
};

void chunk_reduction_algorithm::local_chunk_reduction() {
    for (int cur_dim = gpu_matrix.get_max_dim(); cur_dim >= 1; cur_dim--) {

    }
}

