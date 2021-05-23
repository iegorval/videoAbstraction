#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <windows.h>
#include "videoAbstractor.cuh"

#define FRAMES_STEP 3


std::string dirnameOf(const std::string& fname)
{
    size_t pos = fname.find_last_of("\\/");
    return (std::string::npos == pos) ? "" : fname.substr(0, pos);
}

int main(int argc, char* argv[])
{
    std::string inputPath(argv[1]);
    std::vector<cv::String> fn;

    WCHAR workingPath[MAX_PATH];
    GetModuleFileNameW(NULL, workingPath, MAX_PATH);
    char workingPathChar[MAX_PATH], DefChar = ' ';
    WideCharToMultiByte(CP_ACP, 0, workingPath, -1, workingPathChar, 260, &DefChar, NULL);
    std::string workingPathString(workingPathChar);
    std::string workingPathDir = dirnameOf(dirnameOf(dirnameOf(workingPathString)));

    cv::glob(workingPathDir + "/" + inputPath + "*.MOV*", fn, false);
    size_t count = fn.size();

    VideoAbstractor videoAbstractor(3.0, 4.25, 2.0);
    for (size_t i = 0; i < count; i++) {
        videoAbstractor.recordVideoAbstraction(fn[i], FRAMES_STEP, true);  
    }
    //for (size_t i = 0; i < 1; i++) {
        //cv::Mat image = cv::imread("input/images/thinking.png");
        //cv::resize(image, image, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        //videoAbstractor.makeAbstractFrame(image);
    //}
    return 0;
}