#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlnne/dlnne.h>


#include "../utils/common_include.h"
#include "../utils/utils.h"
#include "../utils/kernel_function.h"
#include "yolov5.h"




void setParameters(utils::InitParameter& initParameters)
{
	initParameters.class_names = utils::dataSets::coco80;
	initParameters.num_class = 80; // for coco
	initParameters.batch_size = 1;
	initParameters.dst_h = 640;
	initParameters.dst_w  = 640;

	/*initParameters.dst_h = 1280;
	initParameters.dst_w = 1280;*/
	// yolov5.6.0
	// initParameters.input_output_names = { "images",  "output"};
	// yolov5.7.0
	initParameters.input_output_names = { "images",  "output0"}; // see line 141
	initParameters.conf_thresh = 0.25f;
	initParameters.iou_thresh = 0.45f;
	initParameters.save_path = "../output";
	initParameters.topK = 300;
}


void task(YOLOV5& yolo, const utils::InitParameter& param, cv::Mat& img, const int& delayTime, const int& batchi,const bool& isShow, const bool& isSave)
{
	yolo.copy(img);

	utils::DeviceTimer d_t1; 
	yolo.preprocess();  
	float duration_1 = d_t1.getUsedTime();

	auto start_2 = std::chrono::high_resolution_clock::now();
	yolo.infer();
	auto end_2 = std::chrono::high_resolution_clock::now();
	auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2) / 1000.0;
	
	auto start_3 = std::chrono::high_resolution_clock::now();
	yolo.postprocess();
	auto end_3 = std::chrono::high_resolution_clock::now();
	auto duration_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3) / 1000.0;

	std::cout << "Yolov5s 前处理: " << duration_1 << " ms 推理: " << duration_2.count() << " ms 后处理: " << duration_3.count() 
				<< " ms" << std::endl;

	if(isShow)
		utils::show(yolo.getObjectss(), param.class_names, delayTime, img);
	if(isSave)
		utils::save(yolo.getObjectss(), param.class_names, param.save_path, img, batchi);
	yolo.reset();
}


int main(int argc ,char *argv[]) {
    
    cv::CommandLineParser parser(argc, argv,
		{
			"{model 	|| tensorrt model file			  }"
			"{size      || image (h, w), eg: 640		  }"
			"{batch_size|| batch size           		  }"
			"{video     || video's path					  }"
			"{img       || image's path					  }"
			"{cam_id    || camera's device id,eg:0		  }"
			"{show      || if show the result			  }"
			"{savePath  || save path, can be ignore		  }"
			"{version   || v560=yolov5.6.0,v570=yolov5.7.0}"
			"{debug     || if debug this program or show more information}"
		});

    utils::InitParameter param;
    setParameters(param);

    std::string model_path = "/home/firefly/Tingshuo/yolov5s/yolov5s_7sim.quantized.rlym";
    std::string engine_path = "/home/firefly/Tingshuo/yolov5s/yolov5s_7sim.quantized.slz";
    std::string video_path = "/home/firefly/Tingshuo/yolov5s/vedio.mp4";
    std::string image_path = "/home/firefly/Tingshuo/yolov5s/640640.jpg";
    int camera_id = 0;
    utils::InputStream source;
    source = utils::InputStream::VIDEO;
    // source = utils::InputStream::IMAGE;
    int size = 640;
	int batch_size = 1;
	bool is_show = true;
	bool is_save = false;
    std::string yolov5_version = "v570";
    
    {
        if(parser.has("model"))
        {
            model_path = parser.get<std::string>("model");
            std::cout << "model_path = " << model_path << std::endl;
        }
        if(parser.has("size"))
        {
            size = parser.get<int>("size");
            std::cout << "size = " << size << std::endl;
            param.dst_h = param.dst_w = size;
        }
        if(parser.has("batch_size"))
        {
            batch_size = parser.get<int>("batch_size");
            std::cout << "batch_size = " << batch_size << std::endl;
            param.batch_size = batch_size;
        }
        if(parser.has("video"))
        {
            source = utils::InputStream::VIDEO;
            video_path = parser.get<std::string>("video");
            std::cout << "video_path = " << video_path << std::endl;
        }
        if(parser.has("img"))
        {
            source = utils::InputStream::IMAGE;
            image_path = parser.get<std::string>("img");
            std::cout << "image_path = " << image_path << std::endl;
        }
        if(parser.has("cam_id"))
        {
            source = utils::InputStream::CAMERA;
            camera_id = parser.get<int>("cam_id");
            std::cout << "camera_id = " << camera_id << std::endl;
        }
        if(parser.has("show"))
        {
            is_show = true;
            std::cout << "is_show = " << is_show << std::endl;
        }
        if(parser.has("savePath"))
        {
            is_save = true;
            param.save_path = parser.get<cv::String>("savePath");
            std::cout << "save_path = " << param.save_path << std::endl;
        }
        if(parser.has("version"))
        {
            yolov5_version = parser.get<cv::String>("version");
            std::cout << "yolov5 version = " << yolov5_version << std::endl;
            if (yolov5_version=="v570")
            {
                param.input_output_names[0] = "images";  // for yolov5.7.0
                param.input_output_names[1] = "output0"; // note: yolov5.6.0 correspond to "output"
            }
        }
		if (parser.has("debug")) {
    		param.is_debug = true;
		}
    }
    // ./test --batch_size=1 --img=/home/firefly/Tingshuo/yolov5s/people.jpg --savePath=../
	// ./test --batch_size=2  --video=/home/firefly/Tingshuo/yolov5s/vedio.mp4  --show
	int delay_time = 1;
    cv::VideoCapture capture;
	if (!setInputStream(source, image_path, video_path, camera_id, capture, delay_time, param)) {
		std::cout << "read the input data errors!" << std::endl;
		return -1;
	}

    YOLOV5 yolo(param);
    int length = 0;
    std::vector<unsigned char> slz_file = utils::loadModel(model_path, engine_path, length);
	if (slz_file.empty()) {
		std::cout << "slz_file is empty!" << std::endl;
		return -1;
	}
	if(param.is_debug)
		std::cout << "load Model finished" << std::endl;
	if (!yolo.init(slz_file, length)) {
		std::cout << "initEngine() ocur errors!" << std::endl;
		return -1;
	}
	if(param.is_debug)
		std::cout << "Yolov5 Init finished" << std::endl;
	if(param.is_debug)
		yolo.check();

	
    cv::Mat frame;
	int batchi = 0;
	while (capture.isOpened())
	{
		if (source != utils::InputStream::IMAGE) capture.read(frame);
		else frame = cv::imread(image_path);
		task(yolo, param, frame, delay_time, batchi, is_show, is_save);
		batchi++;
	}

    return 0;
}