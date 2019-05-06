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
#include <chrono>

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

//const string baseImagePath = "/home/linaro/DepthVO/model/";
const string baseImagePath = "./weights";
const string tmpLsPath = "/tmp/tmpLsPath";

int ListImages(std::string const &path, std::string const &tmppath, std::vector< std::string> &images) {
    images.clear();
    system( ("ls "+ path + " > " + tmppath ).c_str() );
    std::ifstream fin(tmppath, std::ios::in);
    int count = 0;
    char line[1024];
    while ( fin.getline(line,1000) ) {
        std::string name = line;
        std::string ext = name.substr(name.find_last_of(".") + 1);
        if ((ext == "bin") || (ext == "bit")) {
            images.push_back(name);
            count ++;
        }
    }
    return count;
}

float* getfilefrombin(std::string const &filename, float* &result){
    std::ifstream fin;
    fin.open(filename.c_str(), std::ios::binary);
    //获得文件的大小
	fin.seekg(0, std::ios::end);
	long fsize = fin.tellg();
	std::cout << "file len: " << fsize << " num len: "  <<  fsize/(sizeof(float)) << std::endl;
    fin.seekg(0, std::ios::beg);
    std::cout << "1" << std::endl;
    result = new float[fsize/(sizeof(float))];
    std::cout << "2" << std::endl;
    fin.read(( char * ) result, fsize*sizeof(char));
    std::cout << "3" << std::endl;
        std::cout << result << std::endl;
        std::cout << result[2] << std::endl;
    fin.close();
    return result;
}

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

float* dpuVO(DPUKernel *kernel) {
    assert(kernel);
    string images1 = "/home/linaro/DepthVO/images/0000001101.png";
    string images2 = "/home/linaro/DepthVO/images/0000001102.png";
    DPUTask *task;
    task = dpuCreateTask(kernel, 0);
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
	int num = dpuGetOutputTensorSize(task, OUTPUT_NODE);
    float* result = new float[num];
    dpuGetOutputTensorInCHWFP32(task, OUTPUT_NODE, result, num);
    //out_file(task);
    dpuDestroyTask(task);
    return result;
}

int cpuVO(float* result){
    vector<std::string> files;
    vector<float*> weights;
    ListImages(baseImagePath,tmpLsPath, files);
    for(auto fileN : files){
        std::cout << fileN << std::endl;
        float* weights_temp;
        getfilefrombin("weights/"+fileN, weights_temp);
        std::cout << "4" << std::endl;
        weights.push_back(weights_temp);
    }
    // read data
    int datalen=0;
    float* inputdata = result;
    int layerinsize[3] = {7680,512,512};
    int layeroutsize[3] = {512,512,6};
    float* layerout[3];
    for (int layer=0; layer < 3; layer++){
        layerout[layer] = new float[layeroutsize[layer]];
        float* output_layer = layerout[layer];
        float* input_layer = layer ? layerout[layer - 1] : inputdata;
        float* weight_layer = weights[layer*2];
        float* bias_layer = weights[layer*2 + 1];
        int m_layer = layerinsize[layer];
        int n_layer = layeroutsize[layer];
        std::cout<< "m_layer: " << m_layer << " ;n_layer: " << n_layer <<  std::endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        for(int s=0;s<n_layer;s++){  
            output_layer[s] = 0;
            for(int m=0;m<m_layer;m++){  
                    output_layer[s] += input_layer[m] * weight_layer[s*m_layer + m];
            }
        } 
        for(int s=0;s<n_layer;s++){  
            output_layer[s] += bias_layer[s];
        }
        if (layer != 2)
        {
            for(int s=0;s<n_layer;s++)
            {  
                if(output_layer[s] < 0)
                    output_layer[s] = 0;
            }
        }
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout<<" time cost= "<< (time_used.count() * 1000) <<" ms."<<std::endl;
    }
    ofstream outfile1("fc_0_result.txt", ios::out);
    ofstream outfile2("fc_1_result.txt", ios::out);
    ofstream outfile3("fc_2_result.txt", ios::out);
    if(!outfile1) {
        cerr<<"open outfile erro"<<endl;
        exit(1);
    }
    for(int i=0; i<layeroutsize[0]; i++) {
        outfile1<<layerout[0][i]<<" ";
    }
    outfile1.close();
    if(!outfile3) {
        cerr<<"open outfile erro"<<endl;
        exit(1);
    }
    for(int i=0; i<layeroutsize[2]; i++) {
        outfile3<<layerout[2][i]<<" ";
    }
    outfile3.close();
    if(!outfile2) {
        cerr<<"open outfile erro"<<endl;
        exit(1);
    }
    for(int i=0; i<layeroutsize[1]; i++) {
        outfile2<<layerout[1][i]<<" ";
    }
    outfile2.close();
    delete[] inputdata;
    while (!weights.empty())
    {
        float* weights_rel = weights.back();
        delete[] weights_rel;
        weights.pop_back();
    }
    return 0;
} 

int main(void) {
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Load DPU Kernel for DenseBox neural network
    DPUKernel *kernel = dpuLoadKernel("deployVO");

    // Doing face detection.
    cpuVO(dpuVO(kernel));

    // Destroy DPU Kernel & free resources
    dpuDestroyKernel(kernel);

    // Dettach from DPU driver & release resources
    dpuClose();

    return 0;
}