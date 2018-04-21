#pragma once

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define MASTER 0

enum boolean { FALSE, TRUE };

typedef struct Cluster
{
	double x;
	double y;
	double diameter;
	int numOfGroupPoints;
	int id;
};

typedef struct Point
{
	double x;
	double y;
	double vX;
	double vY;
	int clusterId;
};

Point* copyPointsToGPU(Point* myPointArr, int mySize);
boolean ClassifyWithCudaPreparation(int n, int k, Point *arrPoints, Point *cudaPoints, Cluster *arrClusters);
Cluster* copyClusteresToGPU(Cluster* arrClusters, int k);
void gatherAllPoints(int myid, int myPointSize, Point *myPointsArray, int allPointsSize, Point *allPoints, MPI_Datatype MPI_POINT_TYPE);
void synchronizeClusterCenters(int k, Cluster *arrClusters);
MPI_Datatype createClusterDataType();
MPI_Datatype createPointDataType();
Cluster *kMeansPreparationProcesses(int numprocs, int myid, int n, int k, double intervalT, double dt, int limit, double qm, Point *arrPoints,
	double *bestQuality, double *time);
void scatterAllPoints(int n, Point *arrPoints, int *myPointsSize, Point **myPoints, int myid, int numprocs, MPI_Datatype MPI_POINT_TYPE);
void brodcastAllDataProperties(int *n, int *k, double *intervalT, double *dt, int *limit, double *qm);
boolean classifyEachPointToClustereCentersWithCuda(int n, int k, Point *arrPoints, Cluster *arrClusters);
Point* readFromFile(int *n, int *k, double *time, double *dt, int *limit, double *qm, const char *fileName);
void writeToFile(int k, Cluster *arrCluster, double momentTime, double executionTime, double qualtiy, const char *filename);
void kMeans(int myid, MPI_Datatype MPI_POINT_TYPE, MPI_Datatype MPI_CLUSTER_TYPE, int n, int k, double intervalT,
	double dt, int limit, double qm, Point *arrPoints, Cluster *arrClusters, Point *cudaPoints, int allPointsSize, Point *allPoints, double *bestQuality, double *time);
void kMeansAlgorithm(int myid, int n, int k, int limit, Point *arrPoints, Cluster *arrClusters, Point *cudaPoints,
	MPI_Datatype MPI_POINT_TYPE, MPI_Datatype MPI_CLUSTER_TYPE);
void initialClusters(int myid, int n, int k, Point *arrPoints, Cluster *arrClusters, MPI_Datatype MPI_CLUSTER_TYPE);
void recalculateClusterCenters(int n, int k, Point *arrPoints, Cluster *arrClusters);
void qualityEvaluate(int n, int k, Point *arrPoints, Cluster *arrClusters, double *quality);
void calculateClustersDiameter(int n, int k, Point *arrPoints, Cluster *arrClusters);
double diameter(int n, int k, Point *arrPoints, int clusterId);
void relocationPoints(int n, double time, Point *arrPoints);
void resetClustersData(int k, Cluster *arrClusters);
double distanceClusterToCluster(Cluster c1, Cluster c2);
double distancePointToPoint(Point p1, Point p2);

// Cuda functions prototype
__device__ double cudaDistance(Point p1, Cluster c1);
void checkCudaStatus(cudaError e, const char *message);
__global__ void  findNeasrstClusterCenter(int blockNums, int k, Point *points, Cluster *clusters, boolean *isChange);

