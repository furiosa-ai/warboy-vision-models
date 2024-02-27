#include "pose_decode.hpp"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

#define assertm(exp, msg) assert(((void)msg, exp))

// Integral module of Distribution Focal Loss (DFL)
// Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
float dfl(const float *const x, const uint32_t reg_max)
{
    float *x_softmax = new float[reg_max];
    float softmax_denom = 0.0f;

    for (uint32_t i = 0; i < reg_max; i++)
    {
        const float y = exp(x[i]);
        x_softmax[i] = y;
        softmax_denom += y;
    }

    float coord = 0.0f;
    for (uint32_t i = 0; i < reg_max; i++)
    {
        coord += (x_softmax[i] / softmax_denom) * i;
    }

    delete[] x_softmax;

    return coord;
}

void yolov8_pose_decode_feat(
    const float stride, const float conf_thres, const uint32_t max_boxes,
    const float *const feat_box, const float *const feat_cls, const float *const feat_pose, const uint32_t batch_size,
    const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max, const uint32_t npos,
    float *const out_batch, uint32_t *out_batch_pos)
{

    const uint32_t output_params_per_box = 5 + npos * 3; // box info (5) + skeleton info(17) * (x,y,conf)(3)

    const uint32_t max_out_batch_pos = output_params_per_box * max_boxes;

    const float *feat_box_cur = feat_box;
    const float *feat_cls_cur = feat_cls;
    const float *feat_pose_cur = feat_pose;

    for (uint32_t b = 0; b < batch_size; b++)
    {
        float *const out = out_batch + (b * max_out_batch_pos); // move to batch
        uint32_t *const out_pos_ptr = out_batch_pos + b;        // get pointer to write position
        uint32_t out_pos = *out_pos_ptr;                        // get write position

        for (uint32_t y = 0; y < ny; y++)
        {
            for (uint32_t x = 0; x < nx; x++)
            {
                float conf = 0.0;

                for (uint32_t c = 0; c < nc; c++)
                {
                    const float conf_cur = *feat_cls_cur++;
                    if (conf_cur > conf)
                    {
                        conf = conf_cur;
                    }
                }
                if (conf > conf_thres)
                {
                    assertm(out_pos + output_params_per_box <= max_out_batch_pos, "Reached max number of boxes");

                    // anchors
                    const float ax = x + 0.5f;
                    const float ay = y + 0.5f;

                    const uint32_t pos = x + y * nx + b * (nx * ny);
                    const uint32_t pos_box = pos * 4 * reg_max;
                    const uint32_t pos_pose = pos * 17 * 3;

                    const float left = dfl(&feat_box_cur[pos_box + 0 * reg_max], reg_max);
                    const float top = dfl(&feat_box_cur[pos_box + 1 * reg_max], reg_max);
                    const float right = dfl(&feat_box_cur[pos_box + 2 * reg_max], reg_max);
                    const float bottom = dfl(&feat_box_cur[pos_box + 3 * reg_max], reg_max);

                    const float x1 = (ax - left) * stride;
                    const float y1 = (ay - top) * stride;
                    const float x2 = (ax + right) * stride;
                    const float y2 = (ay + bottom) * stride;

                    // write box
                    out[out_pos + 0] = x1;
                    out[out_pos + 1] = y1;
                    out[out_pos + 2] = x2;
                    out[out_pos + 3] = y2;
                    out[out_pos + 4] = conf;
                    // out[out_pos + 5] = cls_idx;  // int -> float
                    for (uint32_t idx = 0; idx < npos; idx++)
                    {
                        const float pose_x = ((feat_pose_cur[pos_pose + idx * 3 + 0]) * 2.0f + (ax - 0.5f)) * stride;
                        const float pose_y = ((feat_pose_cur[pos_pose + idx * 3 + 1]) * 2.0f + (ay - 0.5f)) * stride;
                        const float pose_conf = (feat_pose_cur[pos_pose + idx * 3 + 2]);
                        out[out_pos + 5 + idx * 3 + 0] = pose_x;
                        out[out_pos + 5 + idx * 3 + 1] = pose_y;
                        out[out_pos + 5 + idx * 3 + 2] = pose_conf;
                    }

                    out_pos += output_params_per_box;
                }
            }
        }
        *out_pos_ptr = out_pos;
    }
}
