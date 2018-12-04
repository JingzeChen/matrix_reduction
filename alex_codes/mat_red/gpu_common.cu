#include "gpu_common.h"

__device__ indx round_up_to_2s(indx number) {
    // Devised by Sean Anderson, Sepember 14, 2001.
    // http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    auto size = number;
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    size++;
    return size;
}
