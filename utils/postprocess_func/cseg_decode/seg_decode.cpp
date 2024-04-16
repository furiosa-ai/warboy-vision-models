#include "seg_decode.hpp"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;
#define assertm(exp, msg) assert(((void)msg, exp))

inline float sigmoid(const float x) {
    return (1.0f / (1.0f + exp(-x)));
}

void yolov8_seg_decode(
    const float *const mask_in, const float *const proto,
    const uint32_t nc, const uint32_t mh, const uint32_t mw, const uint32_t no,
    float *const output, uint32_t *output_pos
){
    const uint32_t output_params_per_out = mh * mw;
    const float *mask_in_cur = mask_in;
    const float *proto_cur = proto;
    for(uint32_t o = 0; o < no; o++){
        float *const out = output + (o * output_params_per_out);

        uint32_t *const out_pos_ptr = output_pos + o;
        uint32_t out_pos = *out_pos_ptr;

        for(uint32_t p = 0; p < output_params_per_out; p++){
            for(uint32_t c = 0; c < nc; c++){
                const uint32_t mask_in_pos = c + o * nc;
                const uint32_t proto_pos = p + c * output_params_per_out;
                const float r = mask_in_cur[mask_in_pos] * proto_cur[proto_pos];
                out[out_pos] += r;
            }
            out[out_pos] = sigmoid(out[out_pos]);
            out_pos += 1;
        }
        *out_pos_ptr = out_pos;
    }
}
