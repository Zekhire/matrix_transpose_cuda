
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <windows.h>

using namespace std;
cudaError_t TransposeNaive(int *Imatrix, int *Nmatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU);
cudaError_t TransposeShared(int *Imatrix, int *Smatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU);
cudaError_t TransposeSharedNoConflict(int *Imatrix, int *Cmatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU);
cudaError_t IsTransposed(int *Omatrix, int *Tmatrix, int size_x, int size_y, bool* good);

int const TILE = 32;


__global__ void IsTransposedGPU(int*Omatrix, int*Tmatrix, int size_x, int size_y, bool *good) {
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < size_x*size_y)
		if (Omatrix[index] != Tmatrix[index])
			good = false;
}

__global__ void TransposeNaiveGPU(int* Imatrix, int* Nmatrix, int size_x, int size_y)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int x = index % size_x;
	int y = index / size_x;
	if (x < size_x)
		if (y < size_y) {
			Nmatrix[x*size_y + y] = Imatrix[y*size_x + x];
		}
}

__global__ void TransposeNaiveGPU2D(int* Imatrix, int* Nmatrix, int size_x, int size_y)
{
	int x = blockIdx.x*TILE + threadIdx.x;
	int y = blockIdx.y*TILE + threadIdx.y;
	if (x < size_x)
		if (y < size_y)
			Nmatrix[x*size_y + y] = Imatrix[y*size_x + x];
}

__global__ void TransposeSharedGPU(int* Imatrix, int* Smatrix, int size_x, int size_y)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int x = index % size_x;
	int y = index / size_x;
	if (x < size_x)
		if (y < size_y) {
			int Tindex = threadIdx.x;
			extern __shared__ int temp[];
			int Tx = Tindex % TILE;
			int Ty = Tindex / TILE;
			temp[Ty* TILE + Tx] = Imatrix[y*size_x + x];
			__syncthreads();
			Smatrix[x*size_y + y] = temp[Ty* TILE + Tx];
		}
}

__global__ void TransposeSharedGPU2D(int* Imatrix, int* Smatrix, int size_x, int size_y)
{
	int x = blockIdx.x*TILE + threadIdx.x;
	int y = blockIdx.y*TILE + threadIdx.y;
	if (x < size_x)
		if (y < size_y) {
			int Tindex = threadIdx.x;
			__shared__ int temp[TILE][TILE];
			int Tx = threadIdx.x;
			int Ty = threadIdx.y;
			temp[Tx][Ty] = Imatrix[y*size_x + x];
			__syncthreads();
			Smatrix[x*size_y + y] = temp[Tx][Ty];
		}
}

__global__ void TransposeSharedNoConflictGPU2D(int* Imatrix, int* Smatrix, int size_x, int size_y)
{
	int x = blockIdx.x*TILE + threadIdx.x;
	int y = blockIdx.y*TILE + threadIdx.y;
	if (x < size_x)
		if (y < size_y) {
			int Tindex = threadIdx.x;
			__shared__ int temp[TILE][TILE + 1];
			int Tx = threadIdx.x;
			int Ty = threadIdx.y;
			temp[Ty][Tx] = Imatrix[y*size_x + x];
			__syncthreads();
			Smatrix[x*size_y + y] = temp[Ty][Tx];
		}
}



void TransposeCPU(int**Imatrix, int**Pmatrix, int size_x, int size_y) {
	for (int i = 0; i < size_y; i++)
		for (int j = 0; j < size_x; j++)
			Pmatrix[j][i] = Imatrix[i][j];
}


void ShowMatrix(int** matrix, int size_x, int size_y, bool large) {
	if (large) {
		size_x = 16;
		size_y = 16;
	}
	for (int i = 0; i < size_y; i++) {
		for (int j = 0; j < size_x; j++) {
			cout << matrix[i][j] << "\t";
		}
		cout << endl;
	}
}

void Zero(int**Pmatrix, int size_x, int size_y) {
	for (int i = 0; i < size_y; i++)
		for (int j = 0; j < size_x; j++)
			Pmatrix[i][j] = 0;;
}

void IsTransposedCPU(int**Omatrix, int**Tmatrix, int size_y, int size_x, bool*good) {
	for (int i = 0; i < size_y; i++)
		for (int j = 0; j < size_x; j++)
			if (Omatrix[i][j] != Tmatrix[i][j])
				*good = false;

}



int main()
{
	int size_x = 10000;
	int size_y = 10000;
	bool large = true;
	bool good = true;
	cudaError_t cudaStatus;
	int **Imatrix = new int*[size_y];	Imatrix[0] = new int[size_x*size_y];
	int **Omatrix = new int*[size_x];	Omatrix[0] = new int[size_y*size_x];
	int **Tmatrix = new int*[size_x];	Tmatrix[0] = new int[size_y*size_x];


	for (int i = 1; i < size_y; i++)
		Imatrix[i] = Imatrix[i - 1] + size_x; //&Imatrix[0][i*size_x];	//<- ten sam efekt

	for (int i = 1; i < size_x; i++) {
		Omatrix[i] = Omatrix[i - 1] + size_y;
		Tmatrix[i] = Tmatrix[i - 1] + size_y;
	}

	for (int i = 0; i < size_y; i++)
		for (int j = 0; j < size_x; j++)
			Imatrix[i][j] = i * size_x + j;


	cout << "MACIERZ WEJSCIOWA" << endl;
	ShowMatrix(Imatrix, size_x, size_y, large);

	int size_x2 = size_y;
	int size_y2 = size_x;
	for (int i = 0; i < size_y2; i++)
		for (int j = 0; j < size_x2; j++)
			Omatrix[i][j] = j * size_y2 + i;


	cout << endl << endl;
	cout << "SPODZIEWANA MACIERZ WYJSCIOWA" << endl;
	ShowMatrix(Omatrix, size_x2, size_y2, large);
	cout << endl << endl;
	Zero(Tmatrix, size_x2, size_y2);
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	//															TRANSPOSE CPU
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	LARGE_INTEGER StartCPU, StopCPU, TimeCPU;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartCPU);

	TransposeCPU(Imatrix, Tmatrix, size_x, size_y);


	QueryPerformanceCounter(&StopCPU);
	TimeCPU.QuadPart = StopCPU.QuadPart - StartCPU.QuadPart;
	TimeCPU.QuadPart = TimeCPU.QuadPart * 1000 / frequency.QuadPart;	//ms


	IsTransposedCPU(Omatrix, Tmatrix, size_x, size_y, &good);
	if (!good) {
		cout << "BLAD TRANSPOZYCJI" << endl;
		return -1;
	}



	cout << "TRANSPOZYCJA CPU'OWA	czas: " << TimeCPU.QuadPart << "ms" << endl;
	//TimeCPU.QuadPart = TimeCPU.QuadPart * 1000;
	ShowMatrix(Omatrix, size_x2, size_y2, large);
	cout << endl << endl;
	Zero(Tmatrix, size_x2, size_y2);
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	//															TRANSPOSE NAIVE
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	///LARGE_INTEGER StartNaive, StopNaive, TimeNaive;
	///QueryPerformanceFrequency(&frequency);
	///QueryPerformanceCounter(&StartNaive);



	cudaStatus = TransposeNaive(Imatrix[0], Tmatrix[0], size_x, size_y, TimeCPU);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "TransposeNaive failed!");
		return 1;
	}

	IsTransposedCPU(Omatrix, Tmatrix, size_x, size_y, &good);
	if (!good) {
		cout << "B£¥D TRANSPOZYCJI" << endl;
		return -1;
	}

	///QueryPerformanceCounter(&StopNaive);
	///TimeNaive.QuadPart = StopNaive.QuadPart - StartNaive.QuadPart;
	///TimeNaive.QuadPart = TimeNaive.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms






	//cout << "MACIERZ NAIVNA	czas: " << TimeNaive.QuadPart << "ms" << endl;
	ShowMatrix(Tmatrix, size_x2, size_y2, large);
	cout << endl << endl;
	Zero(Tmatrix, size_x2, size_y2);
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	//															TRANSPOSE SHARED
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	///LARGE_INTEGER StartShared, StopShared, TimeShared;
	///QueryPerformanceFrequency(&frequency);
	///QueryPerformanceCounter(&StartShared);



	cudaStatus = TransposeShared(Imatrix[0], Tmatrix[0], size_x, size_y, TimeCPU);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "TransposeShared failed!");
		cout << endl << cudaGetErrorString(cudaStatus) << endl;
		return 1;
	}



	///QueryPerformanceCounter(&StopShared);
	///TimeShared.QuadPart = StopShared.QuadPart - StartShared.QuadPart;
	///TimeShared.QuadPart = TimeShared.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms

	IsTransposedCPU(Omatrix, Tmatrix, size_x, size_y, &good);
	if (!good) {
		cout << "B£¥D TRANSPOZYCJI" << endl;
		return -1;
	}



	//cout << "MACIERZ SHAREDOWA	czas: " << TimeShared.QuadPart << "ms" << endl;
	ShowMatrix(Tmatrix, size_x2, size_y2, large);
	cout << endl << endl;
	Zero(Tmatrix, size_x2, size_y2);
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	//													TRANSPOSE SHARED NO CONFLICT
	// O=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====<>=====O
	///LARGE_INTEGER StartSharedNoConflicts, StopSharedNoConflicts, TimeSharedNoConflicts;
	///QueryPerformanceFrequency(&frequency);
	///QueryPerformanceCounter(&StartSharedNoConflicts);



	cudaStatus = TransposeSharedNoConflict(Imatrix[0], Tmatrix[0], size_x, size_y, TimeCPU);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "TransposeShared failed!");
		cout << endl << cudaGetErrorString(cudaStatus) << endl;
		return 1;
	}

	IsTransposedCPU(Omatrix, Tmatrix, size_x, size_y, &good);
	if (!good) {
		cout << "B£¥D TRANSPOZYCJI" << endl;
		return -1;
	}




	///QueryPerformanceCounter(&StopSharedNoConflicts);
	///TimeSharedNoConflicts.QuadPart = StopSharedNoConflicts.QuadPart - StartSharedNoConflicts.QuadPart;
	///TimeSharedNoConflicts.QuadPart = TimeSharedNoConflicts.QuadPart * 1000000 / frequency.QuadPart / 1000;	//ms



	//cout << "MACIERZ SHAREDOWA BEZ KONFLIKTÓW	czas: " << TimeSharedNoConflicts.QuadPart << "ms" << endl;
	ShowMatrix(Tmatrix, size_x2, size_y2, large);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t TransposeNaive(int *Imatrix, int *Nmatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU)
{
	int *dev_I = 0;
	int *dev_N = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_I, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_N, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_I, Imatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_N, Nmatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	dim3 block(TILE, TILE);
	int grid_x = (size_x % TILE == 0) ? size_x / TILE : size_x / TILE + 1;
	int grid_y = (size_y % TILE == 0) ? size_y / TILE : size_y / TILE + 1;
	dim3 grid(grid_x, grid_y);
	// Launch a kernel on the GPU with one thread for each element.
	//TransposeNaiveGPU << <10000, 1024 >> > (dev_I, dev_N, size_x, size_y);

	LARGE_INTEGER StartNaive, StopNaive, TimeNaive;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartNaive);

	TransposeNaiveGPU2D << <grid, block >> > (dev_I, dev_N, size_x, size_y);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&StopNaive);
	TimeNaive.QuadPart = StopNaive.QuadPart - StartNaive.QuadPart;
	TimeNaive.QuadPart = TimeNaive.QuadPart * 1000 / frequency.QuadPart;
	cout << "CZAS NAIVNEJ TRANSPOZYCJI= " << TimeNaive.QuadPart << "ms" << endl;
	cout << "PRZYSPIESZENIE GPU WZGLEDEM CPU= " << (double)TimeCPU.QuadPart / TimeNaive.QuadPart << endl;


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Nmatrix, dev_N, size_x*size_y * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_I);
	cudaFree(dev_N);
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t TransposeShared(int *Imatrix, int *Smatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU)
{
	int *dev_I = 0;
	int *dev_S = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_I, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_S, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_I, Imatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_S, Smatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dim3 block(TILE, TILE);
	int grid_x = (size_x % TILE == 0) ? size_x / TILE : size_x / TILE + 1;
	int grid_y = (size_y % TILE == 0) ? size_y / TILE : size_y / TILE + 1;
	dim3 grid(grid_x, grid_y);
	// Launch a kernel on the GPU with one thread for each element.
	//TransposeSharedGPU << <10000, 1024, TILE*TILE * sizeof(int) >> > (dev_I, dev_S, size_x, size_y, TILE);

	LARGE_INTEGER StartShared, StopShared, TimeShared;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartShared);


	TransposeSharedGPU2D << <grid, block >> > (dev_I, dev_S, size_x, size_y);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&StopShared);
	TimeShared.QuadPart = StopShared.QuadPart - StartShared.QuadPart;
	TimeShared.QuadPart = TimeShared.QuadPart * 1000 / frequency.QuadPart;
	cout << "CZAS SHAREDOWEJ TRANSPOZYCJI= " << TimeShared.QuadPart << "ms" << endl;
	cout << "PRZYSPIESZENIE GPU WZGLEDEM CPU= " << (double)TimeCPU.QuadPart / TimeShared.QuadPart << endl;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Smatrix, dev_S, size_x*size_y * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_I);
	cudaFree(dev_S);
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t TransposeSharedNoConflict(int *Imatrix, int *Cmatrix, int size_x, int size_y, LARGE_INTEGER TimeCPU)
{
	int *dev_I = 0;
	int *dev_C = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_I, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_C, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_I, Imatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_C, Cmatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	int TILE = 32;
	dim3 block(TILE, TILE);
	int grid_x = (size_x % TILE == 0) ? size_x / TILE : size_x / TILE + 1;
	int grid_y = (size_y % TILE == 0) ? size_y / TILE : size_y / TILE + 1;
	dim3 grid(grid_x, grid_y);
	// Launch a kernel on the GPU with one thread for each element.
	//TransposeSharedGPU << <10000, 1024, TILE*TILE * sizeof(int) >> > (dev_I, dev_S, size_x, size_y, TILE);

	LARGE_INTEGER StartSharedNoConflicts, StopSharedNoConflicts, TimeSharedNoConflicts;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&StartSharedNoConflicts);


	TransposeSharedNoConflictGPU2D << <grid, block >> > (dev_I, dev_C, size_x, size_y);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&StopSharedNoConflicts);
	TimeSharedNoConflicts.QuadPart = StopSharedNoConflicts.QuadPart - StartSharedNoConflicts.QuadPart;
	TimeSharedNoConflicts.QuadPart = TimeSharedNoConflicts.QuadPart * 1000 / frequency.QuadPart;
	cout << "CZAS SHAREDOWEJ TRANSPOZYCJI BEZ KONFLIKTÓW= " << TimeSharedNoConflicts.QuadPart << "ms" << endl;
	cout << "PRZYSPIESZENIE GPU WZGLEDEM CPU= " << (double)TimeCPU.QuadPart / TimeSharedNoConflicts.QuadPart << endl;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(Cmatrix, dev_C, size_x*size_y * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_I);
	cudaFree(dev_C);
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t IsTransposed(int *Omatrix, int *Tmatrix, int size_x, int size_y, bool* good)
{
	int *dev_O = 0;
	int *dev_T = 0;
	bool *dev_g = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_O, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_T, size_x*size_y * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_g, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_O, Omatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_T, Tmatrix, size_x*size_y * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_g, &good, sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int Blocks = (size_x*size_y % 1024) ? size_x * size_y / 1024 : size_x * size_y % 1024 + 1;
	// Launch a kernel on the GPU with one thread for each element.
	//TransposeNaiveGPU << <10000, 1024 >> > (dev_I, dev_N, size_x, size_y);


	IsTransposedGPU << <Blocks, 1024 >> > (dev_O, dev_T, size_x, size_y, dev_g);



	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(good, dev_g, sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_O);
	cudaFree(dev_T);
	cudaFree(dev_g);
	return cudaStatus;
}