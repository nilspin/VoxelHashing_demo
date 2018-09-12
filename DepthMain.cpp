#include "Application.h"
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>

using std::cout;
void printCudaDetails();

int main(int argc, char *argv[])
{

	//std::unique_ptr<Application> app = std::make_unique<Application>();
  
  printCudaDetails();
	std::unique_ptr<Application> app(new Application());
	app->run();
	return 0;
}

void printCudaDetails() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    cout<<"cudaGetDeviceCount returned "<<static_cast<int>(error_id)<<"\n"
           << cudaGetErrorString(error_id)<<"\n";
    cout<<"Result = FAIL\n";
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    cout<<"There are no available device(s) that support CUDA\n";
  } else {
    cout<<"Detected "<<deviceCount<<" CUDA Capable device(s)\n";
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    cout<<"\nDevice: "<<dev<<" "<<deviceProp.name<<"\n";

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout<<"  CUDA Driver Version / Runtime Version "<<driverVersion / 1000<<"."<<(driverVersion % 100) / 10
          <<" / "<<runtimeVersion / 1000<<", "<< ((runtimeVersion % 100) / 10)<<"\n",
    cout<<"  CUDA Capability Major/Minor version number: "<<deviceProp.major<<"."<< deviceProp.minor<<"\n\n";
  }
}