#include "pose_decode.hpp"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

#define assertm(exp, msg) assert(((void)msg, exp))

inline float sigmoid(const float x)
{
    return (1.0f / (1.0f + exp(-x)));
}

inline float sigmoid_inv(const float x)
{
    return -log(1.0f / x - 1.0f);
}

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
                    if (out_pos + output_params_per_box > max_out_batch_pos)
                    {
                        break;
                    }
                    // anchors
                    const float ax = x + 0.5f;
                    const float ay = y + 0.5f;

                    const uint32_t pos = x + y * nx + b * (nx * ny);
                    const uint32_t pos_box = pos * 4 * reg_max;
                    const uint32_t pos_pose = pos * npos * 3;

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
                        const float pose_conf = sigmoid(feat_pose_cur[pos_pose + idx * 3 + 2]);
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

void yolov5_pose_decode_feat(
    const float *const anchors, const uint32_t num_anchors, const float stride, const float conf_thres, const uint32_t max_boxes,
    const float *const feat, const uint32_t batch_size, const uint32_t ny, const uint32_t nx, const uint32_t no, const uint32_t npose,
    float *const out_batch, uint32_t *out_batch_pos)
{

    const uint32_t params_per_lm = 3;
    const uint32_t params_per_box = 6 + npose * params_per_lm;
    const uint32_t max_out_batch_pos = params_per_box * max_boxes;

    uint32_t pos = 0;
    const float conf_thres_logit = sigmoid_inv(conf_thres);

    for (uint32_t b = 0; b < batch_size; b++)
    {
        float *const out = out_batch + (b * max_out_batch_pos);
        uint32_t *const out_pos_ptr = out_batch_pos + b;
        uint32_t out_pos = *out_pos_ptr;

        for (uint32_t a = 0; a < num_anchors; a++)
        {
            const float ax = anchors[2 * a + 0] * stride;
            const float ay = anchors[2 * a + 1] * stride;

            for (uint32_t y = 0; y < ny; y++)
            {
                for (uint32_t x = 0; x < nx; x++)
                {

                    const float *const feat_cur = &feat[pos];
                    const float conf_obj_logit = feat_cur[4];

                    // early stopping
                    if (conf_obj_logit >= conf_thres_logit)
                    {
                        if (out_pos + params_per_box > max_out_batch_pos)
                        {
                            break;
                        }
                        float conf = sigmoid(feat_cur[5]);
                        conf *= sigmoid(conf_obj_logit);

                        if (conf > conf_thres)
                        {
                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            float bx = sigmoid(feat_cur[0]);
                            float by = sigmoid(feat_cur[1]);
                            float bw = sigmoid(feat_cur[2]);
                            float bh = sigmoid(feat_cur[3]);

                            bx = (bx * 2.0f - 0.5f + x) * stride;
                            by = (by * 2.0f - 0.5f + y) * stride;

                            bw *= 2.0f;
                            bh *= 2.0f;
                            bw = (bw * bw) * ax;
                            bh = (bh * bh) * ay;

                            // xywh -> xyxy
                            const float bw_half = 0.5f * bw;
                            const float bh_half = 0.5f * bh;

                            const float bx1 = bx - bw_half;
                            const float bx2 = bx + bw_half;
                            const float by1 = by - bh_half;
                            const float by2 = by + bh_half;

                            out[out_pos + 0] = bx1;
                            out[out_pos + 1] = by1;
                            out[out_pos + 2] = bx2;
                            out[out_pos + 3] = by2;
                            out[out_pos + 4] = conf;
                            out[out_pos + 5] = 0;

                            // lm
                            for (uint32_t k = 0; k < npose; k++)
                            {
                                const uint32_t kidx = 6 + k * 3;
                                float kx = feat_cur[kidx + 0];
                                float ky = feat_cur[kidx + 1];
                                float kconf = feat_cur[kidx + 2];

                                kx = (kx * 2.0f - 0.5f + x) * stride;
                                ky = (ky * 2.0f - 0.5f + y) * stride;
                                kconf = sigmoid(kconf);

                                const uint32_t kidx_out = out_pos + 6 + k * 3;
                                out[kidx_out + 0] = kx;
                                out[kidx_out + 1] = ky;
                                out[kidx_out + 2] = kconf;
                            }
                            // move one box forward to next write position
                            out_pos += params_per_box;
                        }
                    }
                    pos += no;
                }
            }
        }
        // update buffer end
        *out_pos_ptr = out_pos;
    }
}
