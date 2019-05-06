#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

/* header file for DNNDK APIs */
#include <dnndk/dnndk.h>

#define INPUT_NODE "conv_0_pose"
#define OUTPUT_NODE "conv_5_pose"
#define length 4096

using namespace std;
//using namespace std::chrono;
using namespace cv;

const string baseImagePath = "/home/linaro/DepthVO/model/";

void out_file(DPUTask* task) {
    int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
    float* result = new float[num];
    dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE, result, num);
    //result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
    ofstream outfile("result_HWC.txt", ios::out);
    if(!outfile) {
            cerr<<"open outfile erro"<<endl;
            exit(1);
    }
    for(int i=0; i<num; i++) {
            outfile<<result[i]<<" ";
    }
    outfile.close();

    dpuGetOutputTensorInCHWFP32(task, OUTPUT_NODE, result, num);
    //result = dpuGetOutputTensorAddress(task, OUTPUT_NODE);
    ofstream outfile1("result_CHW.txt", ios::out);
    if(!outfile1) {
            cerr<<"open outfile erro"<<endl;
            exit(1);
    }
    for(int i=0; i<num; i++) {
            outfile1<<result[i]<<" ";
    }
    outfile1.close();
    delete[] result;
}


void VO(DPUKernel *kernel) {
    assert(kernel);
    string images1 = "/home/linaro/DepthVO/images/0000001101.png";
    string images2 = "/home/linaro/DepthVO/images/0000001102.png";
    DPUTask *task;
    task = dpuCreateTask(kernel, 0);
	//int channel = dpuGetOutputTensorChannel(task, OUTPUT_NODE);
	//int8_t *Result = new int8_t[channel];
	DPUTensor* Result;

	Mat image1 = imread(images1);
    Mat image2 = imread(images2);

	resize(image1, image1, Size(608,160), (0, 0), (0, 0), INTER_LINEAR);
    resize(image2, image2, Size(608,160), (0, 0), (0, 0), INTER_LINEAR);
	// mean
    cout << float(image1.at<Vec3b>(0, 0)[0]) << endl;
    cout << float(image2.at<Vec3b>(0, 0)[0]) << endl;
	/*for (int row = 0; row < height; row++) 
    {
		for (int col = 0; col < width; col++) 
        {
            int b = image.at<Vec3b>(row, col)[0];
            int g = image.at<Vec3b>(row, col)[1];
            int r = image.at<Vec3b>(row, col)[2];
            image.at<Vec3b>(row, col)[0] = b - 104;
            image.at<Vec3b>(row, col)[1] = g - 117;
            image.at<Vec3b>(row, col)[2] = r - 123;
        }
    }*/
    //cout << float(image.at<Vec3b>(0,0)[0]) << endl;
    //cout << float(image.at<Vec3b>(0,0)[0]) - 104 << endl;
    // 3 -> 6
    float* data = new float [583680];
    memset((void*)data, 0, 583680*4);
    /*for (int i = 0; i < 583680; i++)
    {
        data[i] = 128;
    }*/
    //cout << float(image.at<Vec3b>(1, 1)[1]) << endl;
    //cout << img.at<Vec6f>(1, 1)[1] << endl;
    for (int c = 0; c < 6; c++)
    {
        for (int h = 0; h < 160; h++)
        {
            for (int w = 0; w < 608; w++)
            {
                if (c == 0)
                data[c*160*608 + h*608 + w] = float(image2.at<Vec3b>(h, w)[0]) - 104;
                if (c == 1)
                data[c*160*608 + h*608 + w] = float(image2.at<Vec3b>(h, w)[1]) - 117;
                if (c == 2)
                data[c*160*608 + h*608 + w] = float(image2.at<Vec3b>(h, w)[2]) - 123;
                if (c == 3)
                data[c*160*608 + h*608 + w] = float(image1.at<Vec3b>(h, w)[0]) - 104;
                if (c == 4)
                data[c*160*608 + h*608 + w] = float(image1.at<Vec3b>(h, w)[1]) - 117;
                if (c == 5)
                data[c*160*608 + h*608 + w] = float(image1.at<Vec3b>(h, w)[2]) - 123;
            }
        }
    }
    //cout << data[0] << ' ' << data[1] << endl;
    //cout << data[160*608] << ' ' << data[160*608+1] << endl;
    //cout << data[608] << ' ' << data[609] << endl;
	dpuSetInputTensorInCHWFP32(task, INPUT_NODE, data, 583680);
	dpuRunTask(task);
	Result = dpuGetOutputTensor(task, OUTPUT_NODE);
    out_file(task);
    dpuDestroyTask(task);
}

int main(void) {
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Load DPU Kernel for DenseBox neural network
    DPUKernel *kernel = dpuLoadKernel("deployVO");

    // Doing face detection.
    VO(kernel);

    // Destroy DPU Kernel & free resources
    dpuDestroyKernel(kernel);

    // Dettach from DPU driver & release resources
    dpuClose();

    return 0;
}