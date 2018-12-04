//
// Created by 唐艺峰 on 2018/7/31.
//

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

__device__ int count_bit_sets(bits64 i) {
    // By Maciej Hehl
    // https://stackoverflow.com/questions/2709430/count-number-of-bits-in-a-64-bit-long-big-integer
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (int)((((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56);
}

__device__ indx most_significant_bit_pos(bits64 v) {
    unsigned int r;
    unsigned int shift;
    r =     (unsigned int) (v >  0xFFFFFFFF) << 5; v >>= r;
    shift = (unsigned int) (v > 0xFFFF) << 4; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (unsigned int) (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return BLOCK_BITS - 1 - r;
}

__device__ indx least_significant_bit_pos(bits64 v) {
    v &= (~v + 1);
    return most_significant_bit_pos(v);
}