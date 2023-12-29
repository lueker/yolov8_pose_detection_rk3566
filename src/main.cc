// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include <dirent.h>

double __get_us(struct timeval t) 
{ 
    return (t.tv_sec * 1000000 + t.tv_usec); 
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint8_t* pMaxClass, uint8_t outputCount, uint8_t topNum)
{
    uint8_t i, j;
    #define MAX_TOP_NUM 20

    if (topNum > MAX_TOP_NUM)
    {
        return 0;
    }
    

    memset(pfMaxProb, 0, sizeof(float) * topNum);
    memset(pMaxClass, 0xff, sizeof(float) * topNum);

    for (j = 0; j < topNum; j++) 
    {
        for (i = 0; i < outputCount; i++) 
        {
            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
                (i == *(pMaxClass + 4))) 
            {
                continue;
            }

            if (pfProb[i] > *(pfMaxProb + j)) 
            {
                *(pfMaxProb + j) = pfProb[i];
                *(pMaxClass + j) = i;
            }
        }
    }

    return 1;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int detect(unsigned char *model_data, int model_data_size, cv::Mat &src_image, char *save_image_path, rknn_context &ctx, void *resize_buf, std::vector<float> &DetectiontRects, std::vector<float> &DetectKeyPoints)
{
    int ret = 0;
    struct timeval start_time, stop_time;

    // RGA
    rga_buffer_t src;
    rga_buffer_t dst;
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    
    cv::Mat img;
    cv::cvtColor(src_image, img, cv::COLOR_BGR2RGB);

    int img_width = img.cols;
    int img_height = img.rows;

    printf("img width = %d, img height = %d\n", img_width, img_height);

    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    
    // 对输入图像用RGA做resize
    if (img_width != width || img_height != height)
    {
        printf("resize with RGA!\n");
        resize_buf = malloc(height * width * channel);
        memset(resize_buf, 0x00, height * width * channel);

        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf, width, height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret)
        {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
            return -1;
        }
        IM_STATUS STATUS = imresize(src, dst);
        inputs[0].buf = resize_buf;
    }
    else
    {
        inputs[0].buf = (void *)img.data;
    }

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("pose_detection time =  %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 关键点检测后处理
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    int8_t *pblob[9];
    for (int i = 0; i < io_num.n_output; ++i)
    {
        pblob[i] = (int8_t *)outputs[i].buf;
    }


    GetResultRectYolov8 PostProcess;
    PostProcess.GetConvDetectionResult(pblob, out_zps, out_scales, DetectiontRects, DetectKeyPoints);

    return 0;
}

int pose_clssifacation(unsigned char *model_data, int model_data_size, rknn_context &ctx, std::vector<float> &pose_points, uint8_t &pose_class)
{
    int ret;
    struct timeval start_time, stop_time;

    
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) 
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) 
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) 
    {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) 
    {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }
    
    // 设置输入数据（重要）
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_FLOAT32;
    inputs[0].size  = 24* sizeof(float);
    inputs[0].fmt   = RKNN_TENSOR_NCHW;
    inputs[0].buf   = &pose_points;
    inputs[0].pass_through = 0;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) 
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // 推理
    gettimeofday(&start_time, NULL); //推理开始时间戳
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) 
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // 获取输出
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);

    if (ret < 0) 
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // Post Process
    for (int i = 0; i < io_num.n_output; i++) 
    {
        uint8_t MaxClass[5];
        float    fMaxProb[5];
        float*   buffer = (float*)outputs[i].buf;
        uint8_t sz     = outputs[i].size / 4;

        rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);

        pose_class = MaxClass[0];

        // printf(" --- Top5 ---\n");
        // for (int i = 0; i < 5; i++) 
        // {
        //     printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
        // }
    }
    gettimeofday(&stop_time, NULL); //推理结束时间戳
    printf("pose_cls infer time =  %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    rknn_outputs_release(ctx, 1, outputs);


    return 0;
}

int main(int argc, char **argv)
{
    char model_path[256] = "./model/yolov8npose.rknn";
    char pose_model_path[256] = "./model/pose_cls.rknn";
    char image_path[256] = "./model/test.jpg";
    char save_image_path[256] = "./test_result.jpg";
    
    rknn_context ctx1 = 0,ctx2 = 0;
    void *resize_buf = nullptr;
    
    // 加载检测模型
    int model_data_size1 = 0;
    unsigned char *model_data1 = load_model(model_path, &model_data_size1);
    
    // 加载分类模型
    int model_data_size2 = 0;
    unsigned char *model_data2 = load_model(pose_model_path, &model_data_size2);
    
    printf("read %s ...\n", image_path);
    cv::Mat src_image = cv::imread(image_path, 1);
    if (!src_image.data)
    {
        printf("cv::imread %s fail!\n", image_path);
        return -1;
    }
    
    int img_width  = src_image.cols;
    int img_height = src_image.rows;
    
    // 将检测结果按照classId、score、xmin1、ymin1、xmax1、ymax1 的格式存放在DetectiontRects中
    std::vector<float> DetectiontRects;

    // 将17个关键点按照每个点（score, x, y）的顺序保存在DetectKeyPoints中
    std::vector<float> DetectKeyPoints;

    detect(model_data1, model_data_size1, src_image, save_image_path, ctx1, resize_buf, DetectiontRects, DetectKeyPoints);
    // printf("detect finish!");
    std::vector<float> pose_points;
    int KeyPointsNum = 17;
    float pose_score = 0;
    int pose_x = 0, pose_y = 0;
    int NumIndex = 0, Temp = 0;
    int xmin_, ymin_;
	
    for (int i = 0; i < DetectiontRects.size(); i += 6)
    {
        int classId = int(DetectiontRects[i + 0]);
        float conf = DetectiontRects[i + 1];
        int xmin = int(DetectiontRects[i + 2] * float(img_width) + 0.5);
        int ymin = int(DetectiontRects[i + 3] * float(img_height) + 0.5);
        int xmax = int(DetectiontRects[i + 4] * float(img_width) + 0.5);
        int ymax = int(DetectiontRects[i + 5] * float(img_height) + 0.5);
        xmin_ = xmin;
        ymin_ = ymin;
        char text1[256];
        sprintf(text1, "%d:%.2f", classId, conf);
        rectangle(src_image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
        putText(src_image, text1, cv::Point(xmin, ymin + 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	
	    Temp = 0; 
        // NumIndex表示a当前图像的第几个目标，Temp用来控制关节点颜色，头部，身体，腿部颜色各不相同
        for (int k = NumIndex * KeyPointsNum * 3; k < (NumIndex + 1)* KeyPointsNum * 3 ; k += 3)
        {
            pose_score = DetectKeyPoints[k + 0];
            
            if(pose_score > 0.5)
            {
                if(k > 5 * 3)
                {   
                    pose_points.push_back(DetectKeyPoints[k + 1]);
                    pose_points.push_back(DetectKeyPoints[k + 2]);
                }
                pose_x = int(DetectKeyPoints[k + 1] * float(img_width) + 0.5);
                pose_y = int(DetectKeyPoints[k + 2] * float(img_height) + 0.5);
                if(Temp < 5)
                {
                    cv::circle(src_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(0, 0, 255), 5);
                }
                else if(5 <= Temp && Temp < 12)
                {
                    cv::circle(src_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(0, 255, 0), 5);
                }
                else
                {
                    cv::circle(src_image, cv::Point(pose_x, pose_y), 2, cv::Scalar(255, 0, 0), 5);
                }
            }
            Temp += 1;
        }
        NumIndex += 1;
    }

    uint8_t pose_class;
    pose_clssifacation(model_data2, model_data_size2, ctx2, pose_points, pose_class);

    char text2[256];
    sprintf(text2, "pose_class: %d", pose_class);
    putText(src_image, text2, cv::Point(xmin_, ymin_ - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    cv::imwrite(save_image_path, src_image);

    printf("== obj: %d \n", int(float(DetectiontRects.size()) / 6.0));
    
    if (resize_buf)
    {
        free(resize_buf);
    }
    
    if (ctx1 > 0 || ctx2 >=0)
    {
        rknn_destroy(ctx1);
        rknn_destroy(ctx2);
    }
    
    if (model_data1 || model_data2) 
    {
        free(model_data1);
        free(model_data2);
    }
    
    return 0;
}
