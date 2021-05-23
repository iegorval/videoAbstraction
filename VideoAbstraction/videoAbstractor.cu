#include "videoAbstractor.cuh"

__constant__ float devKernel[KERNEL_SIZE], devKernelEdges[2 * KERNEL_SIZE_DOG];

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess) {
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

VideoAbstractor::VideoAbstractor(float sigmaSpace, float sigmaRange, float sigmaEdges) {
	this->sigmaSpace = sigmaSpace; this->sigmaRange = sigmaRange;
	precomputeKernel(spaceKernel, KERNEL_RADIUS, sigmaSpace);
	precomputeKernel(edgesKernel, KERNEL_RADIUS_DOG, sigmaEdges);
	precomputeKernel(edgesKernel, KERNEL_RADIUS_DOG, sqrtf(1.6f) * sigmaEdges);
	
	// Save the space kernel to the constant memory
	cudaMemcpyToSymbol(devKernel, spaceKernel.data(), spaceKernel.size() * sizeof(float));
	cudaMemcpyToSymbol(devKernelEdges, edgesKernel.data(), edgesKernel.size() * sizeof(float));
	gpuErrchk(cudaPeekAtLastError());

	// Specify the texture description
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeMirror;
	texDesc.addressMode[1] = cudaAddressModeMirror;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = false;
}

int VideoAbstractor::recordVideoAbstraction(cv::String in_name, int frames_step, bool show) {
	cv::VideoCapture capture(in_name);
	if (!capture.isOpened()) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}
	cv::VideoWriter writer;
	std::string::size_type dot_p = in_name.find_last_of('.');
	std::string out_name = in_name.substr(0, dot_p) + "_abstracted_no_quantization.avi";
	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	writer.open(out_name, codec, capture.get(cv::CAP_PROP_FPS), cv::Size(OUT_WIDTH, OUT_HEIGHT), true);
	int duration, frame_idx = 0, duration_sum = 0, n_processed_frames = 0;
#ifdef LOG_TIME
	std::ofstream flog;
	flog.open("durations_log_no_edges.txt");
	if (!flog.is_open()) std::cout << "Cannot open the file." << std::endl;
#endif
	while (true) {
		cv::Mat frame;
		capture >> frame;
		if (frame.empty()) break;
		if (frame_idx++ % frames_step == 0) {
			cv::resize(frame, frame, cv::Size(OUT_WIDTH, OUT_HEIGHT), 0, 0, cv::INTER_LINEAR);
#ifdef LOG_TIME
			auto start = std::chrono::high_resolution_clock::now();
#endif
			frame = makeAbstractFrame(frame);
#ifdef LOG_TIME
			auto stop = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
			duration_sum += duration; 
			n_processed_frames++;
			flog << std::fixed << std::setprecision(5) << duration << std::endl;
#endif
			for (int i = 0; i < frames_step; i++) writer << frame;
			if (show) cv::imshow("Frame", frame);
		}
		char c = (char)cv::waitKey(25);
		if (c == 27) break;
	}
#ifdef LOG_TIME
	float duration_avg = duration_sum / (float)n_processed_frames;
	std::cout << "Average running time [ms]: " << duration_avg << std::endl;
	flog.close();
#endif
	capture.release();
	cv::destroyAllWindows();
}

void VideoAbstractor::precomputeKernel(std::vector<float>& kernel, int kernelRad, float sigma) {
	float normConst = 1.f / (float)(sqrtf(2 * M_PI) * sigma);
	float expConst = -1.f / (2 * sigma * sigma);
	int kernelSize = 2 * kernelRad + 1;
	for (int i = 0; i < kernelSize; i++) {
		float x = (float)i - kernelRad;
		kernel.push_back(normConst * expf(expConst * x * x));
	}
}

cv::Mat VideoAbstractor::makeAbstractFrame(cv::Mat frame) {
	cv::Mat frameLab, frameFloat, imgEdges;
	frame.convertTo(frameFloat, CV_32F, 1.f / 255.f);
	cv::cvtColor(frameFloat, frameLab, cv::COLOR_BGR2Lab);
	int height = frameFloat.rows, width = frameFloat.cols;
	// Split the image into separate channels
	cv::Mat channelImages[N_CHANNELS];
	cv::split(frameLab, channelImages);
	// Apply the bilateral filtering to abstract the frame
	for (int abstract_it = 0; abstract_it < N_ITERATIONS; abstract_it++) {
#ifdef USE_DOG
		if (abstract_it == N_ITERATIONS_DOG) {
			imgEdges = detectDoGEdges(channelImages[0], width, height);
		}
#endif
		makeAbstractIteration(channelImages);
	}
#ifdef USE_QUANTIZATION 
	// Apply the optional luminance-based quantization
	quantizeLuminance(channelImages[0], width, height);
#endif
#ifdef USE_DOG
	// Combine the quantized abstracted image with the DoG edges
	channelImages[0] = imgEdges.mul(channelImages[0]) - (1.f - imgEdges) * L_MAX; 
#endif
	// Merge the modified channels back to the output frame
	cv::merge(channelImages, 3, frameLab);
	cv::cvtColor(frameLab, frameFloat, cv::COLOR_Lab2BGR);
	frameFloat.convertTo(frame, CV_8UC3, 255.f);
	//cv::imshow("Abstract image", frame);
	//cv::waitKey();
    //cv::destroyAllWindows();
	return frame;
}

void VideoAbstractor::quantizeLuminance(cv::Mat& frameLuminance, int width, int height) {
	float* devLuminance;
	int imageSize = width * height;
	gpuErrchk(cudaMalloc((void**)&devLuminance, imageSize * sizeof(float)));
	gpuErrchk(cudaMemcpy(devLuminance, frameLuminance.ptr(), imageSize * sizeof(float), cudaMemcpyHostToDevice));
	// Apply the luminance pseudo-quantization
	dim3 gridSize = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0), 1);
	dim3 blockSize = dim3(TILE_SIDE, 1);
	applyQuantization <<< gridSize, blockSize >>> (devLuminance, width, height);
	gpuErrchk(cudaMemcpy(frameLuminance.ptr(), devLuminance, imageSize * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaPeekAtLastError()); 
	cudaFree(devLuminance);
}

void VideoAbstractor::prepareResourceDesc(cudaResourceDesc& resDesc, float* devChannel, size_t pitch, int width, int height) {
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
	resDesc.res.pitch2D.devPtr = devChannel;
	resDesc.res.pitch2D.pitchInBytes = pitch;
}

void VideoAbstractor::makeAbstractIteration(cv::Mat (&channelImages)[N_CHANNELS]) {
	int height = channelImages[0].rows, width = channelImages[1].cols;
	int imageSize = height * width;
	size_t pitch;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	float* devChannels[N_CHANNELS], * devFiltered;

	// Create texture memory for each input channel
	cudaResourceDesc resDesc[N_CHANNELS];
	static cudaTextureObject_t texChannels[N_CHANNELS];
	gpuErrchk(cudaMalloc((void**)&devFiltered, N_CHANNELS * imageSize * sizeof(float)));
	for (int i = 0; i < N_CHANNELS; i++) {
		// Create the resource description and texture object
		gpuErrchk(cudaMallocPitch((void**)&devChannels[i], &pitch, width * sizeof(float), height));
		gpuErrchk(cudaMemcpy2D(devChannels[i], pitch, channelImages[i].ptr(), width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
		gpuErrchk(cudaPeekAtLastError());
		prepareResourceDesc(resDesc[i], devChannels[i], pitch, width, height);
		gpuErrchk(cudaCreateTextureObject(&texChannels[i], &resDesc[i], &texDesc, NULL));
	}

	// Apply the bilateral filtering
	dim3 gridSize = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0), height / TILE_SIDE + (int)((height % TILE_SIDE) > 0));
	dim3 blockSize = dim3(TILE_SIDE, TILE_SIDE);
	applyBilateral <<< gridSize, blockSize >>> (devFiltered, texChannels[0], texChannels[1], texChannels[2], width, height, sigmaRange);
	gpuErrchk(cudaPeekAtLastError());

	// Get the output to the host
	for (int i = 0; i < N_CHANNELS; i++) {
		gpuErrchk(cudaMemcpy(channelImages[i].ptr(), devFiltered + i * imageSize, imageSize * sizeof(float), cudaMemcpyDeviceToHost));
	}

	// Free the GPU memory
	for (int i = 0; i < N_CHANNELS; i++) {
		cudaDestroyTextureObject(texChannels[i]);
		cudaFree(devChannels[i]);
	}
	cudaFree(devFiltered);
}

void VideoAbstractor::apply2DGaussian(cudaTextureObject_t& luminanceTex, float* devIntermediate, 
	float* devOutput, float* devLuminance, int width, int height, size_t pitch, dim3 gridSize, 
	dim3 blockSize, cudaResourceDesc resDesc, int kernelOffset) {
	// Separable 1D Gaussian filters instead of 2D Gaussian computation
	apply1DGaussian <<< gridSize, blockSize >>> (luminanceTex, devIntermediate, width, height, pitch, true, kernelOffset);
	resDesc.res.pitch2D.devPtr = devIntermediate;
	gpuErrchk(cudaCreateTextureObject(&luminanceTex, &resDesc, &texDesc, NULL));
	apply1DGaussian <<< gridSize, blockSize >>> (luminanceTex, devOutput, width, height, 0, false, kernelOffset);
	resDesc.res.pitch2D.devPtr = devLuminance;
	gpuErrchk(cudaCreateTextureObject(&luminanceTex, &resDesc, &texDesc, NULL));
	gpuErrchk(cudaPeekAtLastError());
}

cv::Mat VideoAbstractor::detectDoGEdges(cv::Mat& frameLuminance, int width, int height) {
	float* devLuminance, * devIntermediate, * devGaussianTop, * devGaussianBottom;
	int imageSize = width * height; size_t pitch;
	cv::Mat imgEdges = frameLuminance.clone();

	gpuErrchk(cudaMallocPitch((void**)&devLuminance, &pitch, width * sizeof(float), height));
	gpuErrchk(cudaMallocPitch((void**)&devIntermediate, &pitch, width * sizeof(float), height));
	gpuErrchk(cudaMalloc((void**)&devGaussianTop, imageSize * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&devGaussianBottom, imageSize * sizeof(float)));

	// Create the luminance texture
	cudaResourceDesc resDesc;
	cudaTextureObject_t luminanceTex;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	gpuErrchk(cudaMemcpy2D(devLuminance, pitch, frameLuminance.ptr(), width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice));
	gpuErrchk(cudaPeekAtLastError());
	prepareResourceDesc(resDesc, devLuminance, pitch, width, height);
	gpuErrchk(cudaCreateTextureObject(&luminanceTex, &resDesc, &texDesc, NULL));

	// Compute the DoG between two different blur levels
	dim3 gridSize = dim3(width / TILE_SIDE + (int)((width % TILE_SIDE) > 0), height / TILE_SIDE + (int)((height % TILE_SIDE) > 0));
	dim3 blockSize = dim3(TILE_SIDE, TILE_SIDE);
	apply2DGaussian(luminanceTex, devIntermediate, devGaussianTop, devLuminance, width, height, pitch, gridSize, blockSize, resDesc, 0);
	apply2DGaussian(luminanceTex, devIntermediate, devGaussianBottom, devLuminance, width, height, pitch, gridSize, blockSize, resDesc, 1);
	computeEdgeFunction <<< gridSize, blockSize >>> (devGaussianTop, devGaussianBottom, width, height);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy(imgEdges.ptr(), devGaussianTop, imageSize * sizeof(float), cudaMemcpyDeviceToHost));

	// Free GPU memory
	cudaFree(devLuminance);
	cudaFree(devGaussianTop);
	cudaFree(devGaussianBottom);
	cudaFree(devIntermediate);
	cudaDestroyTextureObject(luminanceTex);
	return imgEdges;
}

__global__ void apply1DGaussian(cudaTextureObject_t luminanceTex, float* devBlurred, int width, int height, int pitch, bool isHorizontal, int kernelOffset) {
	int tIdx = threadIdx.x, tIdy = threadIdx.y;
	int row = blockIdx.y * TILE_SIDE + tIdy;
	int col = blockIdx.x * TILE_SIDE + tIdx;
	float* pitchRowPtr;
	if (row < height && col < width) {
		float conv1d = 0.f;
		for (int i = 0; i < KERNEL_SIZE_DOG; i++) {
			int colCurrent = col - isHorizontal * (KERNEL_RADIUS_DOG - i);
			int rowCurrent = row - (1 - isHorizontal) * (KERNEL_RADIUS_DOG - i);
			conv1d += devKernelEdges[kernelOffset * KERNEL_SIZE_DOG + i] * tex2D<float>(luminanceTex, colCurrent, rowCurrent);
		}
		if (isHorizontal) {
			pitchRowPtr = (float*)((char*)devBlurred + row * pitch);
		}
		else {
			pitchRowPtr = devBlurred + row * width;
		}
		pitchRowPtr[col] = conv1d;
	}
}

__global__ void computeEdgeFunction(float* devGaussianTop, float* devGaussianBottom, int width, int height) {
	int tIdx = threadIdx.x, tIdy = threadIdx.y;
	int row = blockIdx.y * TILE_SIDE + tIdy;
	int col = blockIdx.x * TILE_SIDE + tIdx;
	if (row < height && col < width) {
		int idx = row * width + col;
		float dog = devGaussianTop[idx] - TAU * devGaussianBottom[idx];
		if (dog > 0.f) {
			devGaussianTop[idx] = 1; 
		} else {
			devGaussianTop[idx] = 1 + tanhf(dog * FALLOFF);
		}
	}
}

inline __device__ float gaussianRange(float value, float sigma) {
	return std::expf(-(std::powf(value, 2)) / (2 * powf(sigma, 2))) / (2 * M_PI * powf(sigma, 2));
}

__global__ void applyBilateral(float* devFiltered, cudaTextureObject_t texL,
	cudaTextureObject_t texA, cudaTextureObject_t texB, int width, int height, float sigmaRange) {
	int tIdx = threadIdx.x, tIdy = threadIdx.y;
	int row = blockIdx.y * TILE_SIDE + tIdy;
	int col = blockIdx.x * TILE_SIDE + tIdx;

	if (row < height && col < width) {
		//cudaTextureObject_t tex = texChannels[0];
		float valueCentral[N_CHANNELS] = {
			tex2D<float>(texL, col, row),
			tex2D<float>(texA, col, row),
			tex2D<float>(texB, col, row)
		};
		float totalFilteredValue[3] = { 0.f, 0.f, 0.f };
		float normalizationConstant = 0.f;
		for (int i = 0; i < KERNEL_SIZE; i++) {
			for (int j = 0; j < KERNEL_SIZE; j++) {
				int rowCurrent = row - KERNEL_RADIUS + i;
				int colCurrent = col - KERNEL_RADIUS + j;
				float valueCurrent[3] = {
					tex2D<float>(texL, colCurrent, rowCurrent),
					tex2D<float>(texA, colCurrent, rowCurrent),
					tex2D<float>(texB, colCurrent, rowCurrent)
				};
				float rangeDistance = 0.f;
				for (int ch = 0; ch < N_CHANNELS; ch++) {
					rangeDistance += powf(valueCurrent[ch] - valueCentral[ch], 2);
				}
				rangeDistance = sqrtf(rangeDistance);
				float spaceWeight = devKernel[i] * devKernel[j];
				float rangeWeight = gaussianRange(rangeDistance, sigmaRange);
				float totalWeight = spaceWeight * rangeWeight;
				for (int ch = 0; ch < N_CHANNELS; ch++) {
					totalFilteredValue[ch] += totalWeight * valueCurrent[ch];
				}
				normalizationConstant += totalWeight;
			}
		}
		for (int ch = 0; ch < N_CHANNELS; ch++) {
			devFiltered[ch * width * height + row * width + col] = totalFilteredValue[ch] / normalizationConstant;
		}
	}
}

__global__ void quantizationHistogram(float* devLuminance, unsigned int* devHistogram, int width, int height) {
	__shared__ unsigned int sharedHistogram[HISTOGRAM_BINS];
	int col = blockIdx.x * TILE_SIDE + threadIdx.x;
	if (col < width) {
		for (int row = 0; row < height; row++) {
			int binIdx = (int)fmod(devLuminance[row * width + col], (float)BINS_SIZE);
			atomicAdd(&sharedHistogram[binIdx], 1);
		}
		__syncthreads();
	}
	if (col < HISTOGRAM_BINS) {
		atomicAdd(&devHistogram[col], sharedHistogram[col]);
	}
}

__global__ void applyQuantization(float* devLuminance, int width, int height) {
	int col = blockIdx.x * TILE_SIDE + threadIdx.x;
	float binsSize = L_MAX / HISTOGRAM_BINS;
	if (col < width) {
		for (int row = 0; row < height; row++) {
			float lumValue = devLuminance[row * width + col];
			float qNearest = 0.f;
			// Find the closest bin boundary
			for (float q = binsSize; q <= L_MAX; q += binsSize) {
				if (abs(lumValue - q) < abs(lumValue - qNearest)) {
					qNearest = q;
				} else { 
					break; 
				}
			}
			// Calculate the pseudo-quantization
			devLuminance[row * width + col] = qNearest + (BINS_SIZE / 2.f) * tanhf(PHI_Q * (lumValue - qNearest));
		}
	}
}