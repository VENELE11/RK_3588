#pragma once

#include <cstdint>
#include <stdint.h>

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
static int OBJ_CLASS_NUM = 80;
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25

typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} image_rect_t;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

int post_process(const int8_t *outputs[], const letterbox_t *letter_box,
                 const float conf_threshold, const float nms_threshold,
                 object_detect_result_list *od_results, const int model_in_h,
                 const int model_in_w, const int dfl_len, const int32_t zps[], const float scales[], 
                 const int grid_hs[], const int grid_ws[], const int class_nums);
