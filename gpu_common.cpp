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

__device__ indx most_significant_bit_pos(unsigned long long v) {
    unsigned int r;
    unsigned int shift;

    r =     (unsigned int) (v >  0xFFFFFFFF) << 5; v >>= r;
    shift = (unsigned int) (v > 0xFFFF) << 4; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);

    return 63 - r;
}
