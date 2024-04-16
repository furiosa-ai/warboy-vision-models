#include <stdint.h>

extern "C"
{
    void yolov8_seg_decode(
        const float *const mask_in, const float *const proto,
        const uint32_t nc, const uint32_t mh, const uint32_t mw, const uint32_t no,
        float *const output, uint32_t *output_pos);

}
