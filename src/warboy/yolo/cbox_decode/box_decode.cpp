#include "box_decode.hpp"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
#define assertm(exp, msg) assert(((void)msg, exp))
using namespace std;

float dfl(const float *x, const uint32_t reg_max)
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

void yolov8_box_decode_feat(
    const float stride, const float conf_thres, const uint32_t max_boxes,
    const float *const feat_box, const float *const feat_cls, const float *feat_extra, const uint32_t batch_size,
    const uint32_t ny, const uint32_t nx, const uint32_t nc, const uint32_t reg_max, const uint32_t n_extra,
    float *const out_batch, uint32_t *out_batch_pos)
{
    const uint32_t output_params_per_box = 6 + n_extra;

    const uint32_t max_out_batch_pos = output_params_per_box * max_boxes;

    const float *feat_box_cur = feat_box;
    const float *feat_cls_cur = feat_cls;
    const float *feat_extra_cur = feat_extra;

    for (uint32_t b = 0; b < batch_size; b++)
    {
        float *const out = out_batch + (b * max_out_batch_pos);
        uint32_t *const out_pos_ptr = out_batch_pos + b;
        uint32_t out_pos = *out_pos_ptr;

        for (uint32_t y = 0; y < ny; y++)
        {
            for (uint32_t x = 0; x < nx; x++)
            {
                float conf = 0.0;
                uint32_t cls_idx = 0;

                for (uint32_t c = 0; c < nc; c++)
                {
                    const float conf_cur = *feat_cls_cur++;
                    if (conf_cur > conf)
                    {
                        conf = conf_cur;
                        cls_idx = c;
                    }
                }

                if (conf > conf_thres)
                {
                    assertm(out_pos + output_params_per_box <= max_out_batch_pos, "Reached max number of boxes");
                    if (out_pos + output_params_per_box > max_out_batch_pos)
                    {
                        break;
                    }
                    const float ax = x + 0.5f;
                    const float ay = y + 0.5f;

                    const uint32_t pos = x + y * nx + b * (nx * ny);
                    const uint32_t pos_box = pos * 4 * reg_max;

                    const float left = dfl(&feat_box_cur[pos_box + 0 * reg_max], reg_max);
                    const float top = dfl(&feat_box_cur[pos_box + 1 * reg_max], reg_max);
                    const float right = dfl(&feat_box_cur[pos_box + 2 * reg_max], reg_max);
                    const float bottom = dfl(&feat_box_cur[pos_box + 3 * reg_max], reg_max);

                    const float x1 = (ax - left) * stride;
                    const float y1 = (ay - top) * stride;
                    const float x2 = (ax + right) * stride;
                    const float y2 = (ay + bottom) * stride;

                    out[out_pos + 0] = x1;
                    out[out_pos + 1] = y1;
                    out[out_pos + 2] = x2;
                    out[out_pos + 3] = y2;
                    out[out_pos + 4] = conf;
                    out[out_pos + 5] = cls_idx;

                    if (n_extra > 0)
                    {
                        const uint32_t pos_extra = pos * n_extra;
                        for (uint32_t e = 0; e < n_extra; e++)
                        {
                            out[out_pos + 6 + e] = feat_extra_cur[pos_extra + e];
                        }
                    }
                    out_pos += output_params_per_box;
                }
            }
        }
        *out_pos_ptr = out_pos;
    }
}

void yolov5_box_decode_feat(
    const float *const anchors, const uint32_t num_anchors, const float stride, const float conf_thres, const uint32_t max_boxes,
    const float *const feat, const uint32_t batch_size, const uint32_t ny, const uint32_t nx, const uint32_t no,
    float *const out_batch, uint32_t *out_batch_pos)
{
    const uint32_t output_params_per_box = 6;
    const uint32_t max_out_batch_pos = output_params_per_box * max_boxes;
    const uint32_t nc = no - 5;

    const float *cell = feat;

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
                    const float obj_conf = cell[4];

                    if (obj_conf > conf_thres)
                    {
                        assertm(out_pos + output_params_per_box <= max_out_batch_pos, "Reached max number of boxes");
                        if (out_pos + output_params_per_box > max_out_batch_pos)
                        {
                            break;
                        }
                        float conf = 0.0;
                        uint32_t cls_idx = -1;
                        for (uint32_t c = 0; c < nc; c++)
                        {
                            float conf_cur = cell[5 + c];
                            if (conf_cur > conf)
                            {
                                conf = conf_cur;
                                cls_idx = c;
                            }
                        }

                        conf *= obj_conf;
                        if (conf > conf_thres)
                        {
                            // (feat[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                            // (feat[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                            float bx = cell[0];
                            float by = cell[1];
                            float bw = cell[2];
                            float bh = cell[3];

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
                            out[out_pos + 5] = cls_idx;
                            out_pos += output_params_per_box;
                        }
                    }
                    cell += no;
                }
            }
        }
        *out_pos_ptr = out_pos;
    }
}
