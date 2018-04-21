#include "header.h"

__global__ void  findNeasrstClusterCenter(int blockNums, int k, Point *points, Cluster *clusters, boolean *isChange)
{
	
	int j;
	int indexPoints = threadIdx.x + blockIdx.x * blockNums;
	double currentDistance = 0;
	int minIndex = 0; // save min index in the array
	double minDistance = cudaDistance(points[indexPoints], clusters[0]);
	for (j = 1; j < k; j++)
	{
		currentDistance = cudaDistance(points[indexPoints], clusters[j]);
		if (currentDistance < minDistance)
		{
			minDistance = currentDistance;
			minIndex = j;
		}
	}

	//enough one point that moved to chagne it to TRUE
	if (points[indexPoints].clusterId != clusters[minIndex].id)
		*isChange = TRUE;

	points[indexPoints].clusterId = clusters[minIndex].id;
	
}

boolean classifyEachPointToClustereCentersWithCuda(int n, int k, Point *arrPoints, Cluster *arrClusters)
{
	boolean *dev_isChanged;
	boolean isChange = FALSE;
	cudaError_t cudaStatus;
	cudaDeviceProp props;
	int blockNums, threadNum;
	cudaGetDeviceProperties(&props, 0);

	// calculate the number of blocks required in this computer
	blockNums = 1 + n /props.maxThreadsPerBlock;

	// Allocate GPU arrays - arrPoints, arrClusters.
	cudaStatus = cudaMalloc((void**)&dev_isChanged, sizeof(int));
	checkCudaStatus(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(dev_isChanged, &isChange, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaStatus(cudaStatus, "cudaMemcpy failed!");

	threadNum = n / blockNums;

	// Launch a kernel on the GPU with one thread for each element.
	findNeasrstClusterCenter << < blockNums, threadNum >> >(blockNums, k, arrPoints, arrClusters, dev_isChanged);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//checkCudaStatus(cudaStatus, "cudaDeviceSynchronize returned error code");

	cudaStatus = cudaMemcpy(&isChange, dev_isChanged, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaStatus(cudaStatus, "cudaMalloc failed!");

	cudaFree(dev_isChanged);

	return isChange;
}

void checkCudaStatus(cudaError e, const char *message)
{
	// check if the cuda status was ok
	if (e != cudaSuccess)
	{
		printf(message);
		fflush(stdout);
		exit(1);
	}
}

__device__ double cudaDistance(Point p1, Cluster c1)
{
	double deltaX = pow(p1.x - c1.x, 2);
	double deltaY = pow(p1.y - c1.y, 2);
	return sqrt(deltaX + deltaY);
}

