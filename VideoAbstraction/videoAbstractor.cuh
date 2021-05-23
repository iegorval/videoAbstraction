#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cstddef>
#include <opencv2/opencv.hpp>

#define M_PI 3.14159265358979323846
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define N_CHANNELS 3
#define TILE_SIDE 16
#define KERNEL_RADIUS 9
#define KERNEL_RADIUS_DOG 15
#define N_ITERATIONS_DOG 1
#define N_ITERATIONS 3
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)
#define KERNEL_SIZE_DOG (2 * KERNEL_RADIUS_DOG + 1)
#define HISTOGRAM_BINS 14
#define L_MAX 100
#define PHI_Q 3.0
#define FALLOFF 1.0  // [0.75, 5.0]
#define TAU 0.98
#define BINS_SIZE (L_MAX / HISTOGRAM_BINS)
#define USE_DOG
//#define USE_QUANTIZATION 
#define OUT_WIDTH 640
#define OUT_HEIGHT 480
#define LOG_TIME

#pragma once
/**
* Value of the Gaussian function.
*
* Computes the value of the Gaussian function with the standard deviation
* sigma for the range value (i.e. the difference between the two colors in 
* the Lab color space). It is not precomputed due to the wide range of the 
* possible values.
*
* @param kernel - vector to which the computed kernel values are attached.
* @param sigma - standard deviation parameter, which controls the shape of
*                the Gaussian.
*/
inline __device__ float gaussianRange(float value, float sigma);

/**
 * Bilateral filtration.
 *
 * Applies the bilateral filter on the Lab input, passed as three separate
 * textures, each of which corresponds to a separate channel. The result is
 * then stored at the GPU memory location, determined by the devFiltered.
 * The width and height of image are passed to link the individual CUDA
 * threads to the corresponding image pixels. The sigmaRange determined the
 * shape of the Gaussian, used for the color-based weighting. 
 *
 * @param devFiltered - pointer to the GPU memory where the output should
 *                      be stored.
 * @param texL - texture, containing the L channel of the input.
 * @param texA - texture, containing the A channel of the input.
 * @param texB - texture, containing the B channel of the input.
 * @param width - width of the input image.
 * @param height - height of the input image.
 * @paramsigmaRange - standard deviation of the Gaussian, corresponding to 
 *                    the color-based weights in the bilateral filter.
 */
__global__ void applyBilateral(float* devFiltered, cudaTextureObject_t texL, 
	cudaTextureObject_t texA, cudaTextureObject_t texB, int width, int height, float sigmaRange);

/**
 * Computes the color histogram.
 * 
 * Bins all the values of the input image into the bins, number of which is
 * determined by the HISTOGRAM_BINS parameter. Then, computes the occurence
 * values for each histogram bin. This function was used for testing and is
 * unused in the final abstractization pipeline.
 * 
 * @param devLuminance - pointer to the GPU memory where the input luminance
 *                       is stored.
 * @param devHistogram - pointer to the GPU memory where the output values
 *                       of the histogram should be stored.
 * @param width - width of the input image.
 * @param height - height of the input image.
 */
__global__ void quantizationHistogram(
	float* devLuminance, unsigned int* devHistogram, int width, int height);

/**
 * Luminance-based quantization.
 * 
 * Performs the quantization of the luminance channel into the bins, number 
 * of which is defined by the HISTOGRAM_BINS parameter. The width and height 
 * of image are passed to link the individual CUDA threads to the corresponding 
 * image pixels.
 * 
 * @param devLuminance - pointer to the GPU memory where the input luminance
 *                       is stored.
 * @param width - width of the input image.
 * @param height - height of the input image.
 */
__global__ void applyQuantization(float* devLuminance, int width, int height);

/**
 * 1D Gaussian blur.
 *
 * Applies a 1D Gaussian blur with the pre-computed kernel, stored in the 
 * devKernelEdges CUDA constant. The kernelOffset parameter defined which of
 * the kernels, stored in devKernelEdges, will be used for the filtration. The 
 * input luminance is passed as a texture luminanceTex. The width, height and 
 * pitch of image are passed to link the individual CUDA threads to the image
 * pixels. The blur can be either horizontal or vertical, depending on the
 * isHorizontal argument.
 *
 * @param luminanceTex - texture, containing the L channel of the input.
 * @param devBlurred - pointer to the GPU memory where the output of the 1D
 *                     Gaussian blur should be stored.
 * @param width - width of the input image.
 * @param height - height of the input image.
 * @param pitch - pitch of the texture.
 * @param isHorizontal - whether the horizontal (or vertical) blur is performed.
 * @param kernelOffset - offset (in multiplicants of KERNEL_SIZE) for the constant
 *                       devKernelEdges, defining which Gaussian kernel should be
 *                       used for the filtering.
 */
__global__ void apply1DGaussian(cudaTextureObject_t luminanceTex, 
	float* devBlurred, int width, int height, int pitch, bool isHorizontal, int kernelOffset);

/**
 * Edge function computation.
 *
 * Based on the two different Gaussian-blurred images, computes the DoG
 * (Difference-of-Gaussian) value. This value is then used to compute the
 * indicator for the edgeness of a pixel. A special smoother version of the
 * measure is applied for the temporal coherence.
 *
 * @param devGaussianTop - pointer to the GPU memory where the image, blurred
 *                         with the first kernel from devKernelEdges, is stored.
 * @param devGaussianBottom - pointer to the GPU memory where the image, blurred
 *                            with the second kernel from devKernelEdges, is stored.
 * @param width - width of the input image.
 * @param height - height of the input image.
 */
__global__ void computeEdgeFunction(
	float* devGaussianTop, float* devGaussianBottom, int width, int height);

class VideoAbstractor
{
public:
	/**
	* Constructor for the VideoAbstractor.
	* 
	* Precomputes the Gaussian space kernel for the bilateral filtering and 
	* two Gaussian kernels for the DoG edge detector. Stores them in the CUDA 
	* constant (space kernel in devKernel, and Gaussian kernels in devKernelEdges).
	* Also, specifies the texture description for the texDesc parameter.
	* 
	* @param sigmaSpace - standard deviation for the space bilateral filtering.
	* @param sigmaRange - standard deviation for the color bilateral filtering.
	* @param sigmaEdges - standard deviation for the first Gaussian kernel used
	*                     in the DoG edge detector (the second kernel is then
	*                     computed with sqrt(1.6)*sigmaEdges).
	*/
	VideoAbstractor(float sigmaSpace, float sigmaRange, float sigmaEdges);

	/**
	 * Frame abstractization.
	 *
	 * Makes an abstract version of one input BGR frame by the simplified version
	 * of the pipeline, described in the Real-Time Video Abstraction paper:
	 * http://holgerweb.net/PhD/Research/papers/videoabstraction.pdf.
	 *
	 * @param image - input BGR image.
	 */
	cv::Mat makeAbstractFrame(cv::Mat image);

	int recordVideoAbstraction(cv::String in_name, int frames_step, bool show);

private:
	cudaTextureDesc texDesc; /*!< Description for the CUDA texture. */
	std::vector<float> spaceKernel; /*!< Kernel for the space-based bilateral filtering. */
	std::vector<float> edgesKernel; /*!< Kernel for the DoG Gaussian filtering. */
	float sigmaSpace; /*!< Standard devitation for the space-based bilateral filtering. */
	float sigmaRange; /*!< Standard devitation for the value-based bilateral filtering. */
	float sigmaEdges; /*!< Standard devitation for the DoG Gaussian filtering. */

	/**
	 * Gaussian kernel computation.
	 * 
	 * Precomputes the Gaussian kernel of the length defined by the KERNEL_RADIUS.
	 * The kernel elements are added to the kernel vector, passed as the function 
	 * argument. The sigma parameter is used to determine the shape of the kernel.
	 *
	 * @param kernel - vector to which the computed kernel values are attached.
	 * @param kernelRad - radius of the computed kernel.
	 * @param sigma - standard deviation paremeter, which controls the shape of
	 *                the Gaussian.
	 */
	void precomputeKernel(std::vector<float>& kernel, int kernelRad, float sigma);

	/**
	 * 
	 * Definition of the CUDA resource description.
	 * 
	 * Fills the parameters of the CUDA resource description, passed as the parameter
	 * resDesc. Sets the description type to the 2D pitch, and sets the data pointer
	 * to the corresponding 2D pitch devChannel. Also, sets the width, height and pitch
	 * parameters of the resource description.
	 * 
	 * @param resDesc - CUDA resource description, whose parameters should be filled.
	 * @param devChannel - pointer to the data contents of the texture.
	 * @param pitch - pitch of the texture.
	 * @param width - width of the input image.
	 * @param height - height of the input image.
	 */
	void prepareResourceDesc(
		cudaResourceDesc& resDesc, float* devChannel, size_t pitch, int width, int height);

	/**
	 * A single abstractization iteration.
	 * 
	 * Performs one iteration of the abstractization (i.e. bilateral filtering) on 
	 * the input image in the Lab format. The filtration is done separately for each 
	 * channel, though all three Lab color components are considered when calculating 
	 * the perceptual distance between the two pixels.
	 * 
	 * @param channelImages - input image as the array of its individual channels.
	 */
	void makeAbstractIteration(cv::Mat (&channelImages)[N_CHANNELS]);

	/**
	 * Quantization of the luminance component.
	 * 
	 * Performs the quantization of the image, based on its luminance component.
	 * Separated the luminance values into bins, number of which is defined by the
	 * HISTOGRAM_BINS parameter.
	 * 
	 * @param frameLuminance - image, containing the Lab luminance channel.
	 * @param width - width of the input image.
	 * @param height - height of the input image.
	 */
	void quantizeLuminance(cv::Mat& frameLuminance, int width, int height);

	/**
	 * Difference-of-Gaussian edge detection.
	 * 
	 * Detects the edges in the input image, based on the luminance component. The
	 * edges are determined based on the Difference-of-Gaussian (DoG) values. Two
	 * Gaussian-blurred versions of the input luminance are computed, with varying
	 * standard deviation parameter. Those two resulting images are then subtracted
	 * to get the Difference-of-Gaussian value. A special smoothed function is then
	 * used to compute the edge indicator to increase the temporal coherence.
	 * 
	 * @param frameLuminance - image, containing the Lab luminance channel.
	 * @param width - width of the input image.
	 * @param height - height of the input image.
	 */
	cv::Mat detectDoGEdges(cv::Mat& frameLuminance, int width, int height);
	
	/**
	 * Application of 2D Gaussian blur.
	 * 
	 * Performs a 2D Gaussian blur on the image as a series of two consecutive 1D
	 * Gaussian filters. The input is passed as a CUDA texture object luminanceTex.
	 * The other parameters define the width, height and pitch of the data, and
	 * pointers to the GPU locations of the original input, intermediate output
	 * and the final output of the 2D Gaussian blur. The kernelOffset defines the
	 * offset (in terms of multiplicants of KERNEL_SIZE) inside the devKernelEdges
	 * constant that contains a kernel for each level of the Gaussian blur needed
	 * for the DoG edge detector. The resDesc contains the resource description for
	 * the input texture.
	 * 
	 * @param luminanceTex - texture object, containing the Lab luminance channel.
	 * @param devIntermediate - pointer to the GPU memory where the intermediate
	 *                          result (after one 1D blur) should be stored.
	 * @param devOutput - pointer to the GPU memory where the output should be stored.
	 * @param devLuminance - pointer to the GPU memory where the original input is stored.
	 * @param width - width of the input image.
	 * @param height - height of the input image.
	 * @param pitch - pitch of the texture.
	 * @param gridSize - size of the CUDA grid, with which the apply1DGaussian kernel
	 *                   will be called.
	 * @param blockSize - size of the CUDA block, with which the apply1DGaussian kernel
	 *                    will be called.
	 * @param resDesc - resource description for the input texture.
	 * @param kernelOffset - offset (in multiplicants of KERNEL_SIZE) for the constant
	 *                       devKernelEdges, defining which Gaussian kernel should be
	 *                       used for the filtering.
	 */
	void apply2DGaussian(cudaTextureObject_t& luminanceTex, float* devIntermediate, 
		float* devOutput, float* devLuminance, int width, int height, size_t pitch,
		dim3 gridSize, dim3 blockSize, cudaResourceDesc resDesc, int kernelOffset);
};

