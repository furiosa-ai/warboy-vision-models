#include <stdint.h>

extern "C"
{
    void yolov8_box_decode_feat(
        const float stride, const float conf_thres, const uint32_t max_boxes,
        const float *const feat_box, const float *const feat_cls, const float *const feat_extra,
        const uint32_t batch_size,
        const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max, const uint32_t n_extra,
        float *const out_batch, uint32_t *out_batch_pos);

    void yolov5_box_decode_feat(
        const float *const anchors, const uint32_t num_anchors, const float stride, const float conf_thres, const uint32_t max_boxes,
        const float *const feat, const uint32_t batch_size, const uint32_t ny, const uint32_t nx, const uint32_t no,
        float *const out_batch, uint32_t *out_batch_pos);
}
