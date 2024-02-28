#include <stdint.h>

extern "C"
{
    void yolov8_pose_decode_feat(
        const float stride, const float conf_thres, const uint32_t max_boxes,
        const float *const feat_box, const float *const feat_cls, const float *const feat_pose,
        const uint32_t batch_size,
        const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max, const uint32_t npose,
        float *const out_batch, uint32_t *out_batch_pos);
}