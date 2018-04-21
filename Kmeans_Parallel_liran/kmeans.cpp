#include "header.h"

//Preparation All processes for kmeans:
//All create point and cluster mpi type
//Master brodcast to all processes properties that he read from file
//Master divides points array to parts - and scatter points to all processes
//All copy points array to cuda array
Cluster *kMeansPreparationProcesses(int numprocs, int myid, int n, int k, double intervalT, double dt, int limit, double qm, Point *arrPoints, double *bestQuality, double *time)
{
	int myPointsSize = 0;
	Point *myPoints = NULL;
	Cluster *arrClusters = NULL;
	Point *cudaPoints = NULL;

	// create structs to mpi
	MPI_Datatype MPI_POINT_TYPE = createPointDataType();
	MPI_Datatype MPI_CLUSTER_TYPE = createClusterDataType();

	//brodcast the data properties for all process
	brodcastAllDataProperties(&n,&k, &intervalT, &dt, &limit, &qm);
	//master divide to all processes point array
	scatterAllPoints(n, arrPoints, &myPointsSize, &myPoints, myid, numprocs, MPI_POINT_TYPE);
	cudaPoints = copyPointsToGPU(myPoints, myPointsSize);
	// allocation k clusters array for all
	arrClusters = (Cluster*)calloc(sizeof(Cluster), k);
	if (arrClusters == NULL) {
		printf("Clusters allocation error\n");
		exit(2);
	}
	kMeans(myid, MPI_POINT_TYPE, MPI_CLUSTER_TYPE,myPointsSize, k, intervalT, dt, limit, qm, myPoints, arrClusters,cudaPoints,n,arrPoints ,bestQuality, time);

	cudaFree(cudaPoints);
	free(myPoints);
	return arrClusters;
}

Point* copyPointsToGPU(Point* myPointArr, int mySize)
{
	cudaError cudaStatus;
	Point* cudaPoints;

	cudaStatus = cudaMalloc((void**)&cudaPoints, mySize * sizeof(Point));
	checkCudaStatus(cudaStatus, "cudaMalloc failed!");

	cudaStatus = cudaMemcpy(cudaPoints, myPointArr, mySize * sizeof(Point), cudaMemcpyHostToDevice);
	checkCudaStatus(cudaStatus, "cudaMemcpy failed!");

	return cudaPoints;
}

//Master divide points to all processes
void scatterAllPoints(int n, Point *arrPoints, int *myPointsSize, Point **myPoints, int myid, int numprocs, MPI_Datatype MPI_POINT_TYPE)
{
	int i, div, remain;

	div = n / numprocs;
	remain = n % numprocs;
	// master add the remain to his part arr 
	if (myid == MASTER)
		*myPointsSize = div + remain;
	else
		*myPointsSize = div;
	// allocation array to each process with his size
	*myPoints = (Point*)calloc((*myPointsSize), sizeof(Point));
	if (myid == MASTER)
	{
		// assign the masters points part
		#pragma omp parallel for
		for (i = 0; i < *myPointsSize; i++)
			(*myPoints)[i] = arrPoints[i];

		// rest of the points divide to other processes
		int counterPointProcess = 0;
		int currentProcess = 1; 
		for (i = *myPointsSize; i < n; i++)
		{
			if (counterPointProcess == div) // check if done the point part of the current process
			{
				currentProcess++;
				counterPointProcess = 0;
			}
			MPI_Send(&arrPoints[i], 1, MPI_POINT_TYPE, currentProcess, 0, MPI_COMM_WORLD);
			counterPointProcess++;
		}
	}
	else {
		MPI_Status status;
		for (i = 0; i < *myPointsSize; i++)
			MPI_Recv(&(*myPoints)[i], 1, MPI_POINT_TYPE, MASTER, 0, MPI_COMM_WORLD, &status);
	}

}

//Master brodcast to All given data properties  
void brodcastAllDataProperties(int *n ,int *k, double *intervalT, double *dt, int *limit, double *qm)
{
	MPI_Bcast(n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(k, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(intervalT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(limit, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(qm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
//Create mpi point type
MPI_Datatype createPointDataType()
{
	Point point;
	//MPI_TYPE variables
	MPI_Datatype PointMPIType;
	MPI_Datatype type[5] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE ,MPI_INT };
	int blocklen[5] = { 1, 1, 1 , 1 , 1 };
	MPI_Aint disp[5];
	// Create MPI user data type for point
	disp[0] = (char *)&point.x - (char *)&point;
	disp[1] = (char *)&point.y - (char *)&point;
	disp[2] = (char *)&point.vX - (char *)&point;
	disp[3] = (char *)&point.vY - (char *)&point;
	disp[4] = (char *)&point.clusterId - (char *)&point;
	MPI_Type_create_struct(5, blocklen, disp, type, &PointMPIType);
	MPI_Type_commit(&PointMPIType);
	return PointMPIType;
}
//Create mpi cluster type
MPI_Datatype createClusterDataType()
{
	Cluster cluster;
	//MPI_TYPE variables
	MPI_Datatype ClusterMPIType;
	MPI_Datatype type[5] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE , MPI_INT ,MPI_INT };
	int blocklen[5] = { 1, 1, 1 , 1 , 1 };
	MPI_Aint disp[5];
	// Create MPI user data type for point
	disp[0] = (char *)&cluster.x - (char *)&cluster;
	disp[1] = (char *)&cluster.y - (char *)&cluster;
	disp[2] = (char *)&cluster.diameter - (char *)&cluster;
	disp[3] = (char *)&cluster.numOfGroupPoints - (char *)&cluster;
	disp[4] = (char *)&cluster.id - (char *)&cluster;
	MPI_Type_create_struct(5, blocklen, disp, type, &ClusterMPIType);
	MPI_Type_commit(&ClusterMPIType);
	return ClusterMPIType;

}

// Find the group of cluster that generate the best quality in range of given [0-T].
// Updates arrClusters,clusterId for each point 
void kMeans(int myid, MPI_Datatype MPI_POINT_TYPE, MPI_Datatype MPI_CLUSTER_TYPE,int n, int k, double intervalT, 
	double dt, int limit, double qm, Point *arrPoints, Cluster *arrClusters,Point *cudaPoints,int allPointsSize,Point *allPoints, double *bestQuality, double *time)
{
	double i=0;
	boolean reachToQuality = FALSE;
	printf("Start kmeans!\n");
	fflush(stdout);
	while((i < intervalT) && (!reachToQuality)) 
	{
		// all the processes
	    kMeansAlgorithm(myid,n, k, limit, arrPoints, arrClusters,cudaPoints,MPI_POINT_TYPE,MPI_CLUSTER_TYPE);
		// master gather all points .
		gatherAllPoints(myid,n,arrPoints,allPointsSize,allPoints, MPI_POINT_TYPE);
		if (myid == MASTER)
		{
			qualityEvaluate(allPointsSize, k, allPoints, arrClusters, bestQuality);
			if (*bestQuality <= qm)
				reachToQuality = TRUE;
			
		}
		// notify all slaves if we reached to requested quality
		MPI_Bcast(&reachToQuality, 1, MPI_INT, 0, MPI_COMM_WORLD);
		i++;
		if (!reachToQuality)
		{
			*time = i*dt;
			relocationPoints(n, *time, arrPoints);
			cudaPoints = copyPointsToGPU(arrPoints, n);
		}
		
	}
}

//Master gather from all processes the points 
void gatherAllPoints(int myid,int myPointSize,Point *myPointsArray,int allPointsSize ,Point *allPoints,MPI_Datatype MPI_POINT_TYPE)
{
	int i;
	if (myid == MASTER)
	{
		// assign master point to allPoints array
		#pragma omp parallel for 
		for (i = 0; i < myPointSize; i++)
			allPoints[i] = myPointsArray[i];

		MPI_Status status;
		// gather rest of the points from salves
		for (i = myPointSize; i < allPointsSize; i++)
			MPI_Recv(&allPoints[i], 1, MPI_POINT_TYPE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	}
	else
	{//send slaves points to the master.
		for (i = 0 ; i < myPointSize; i ++)
			MPI_Send(&myPointsArray[i], 1, MPI_POINT_TYPE, MASTER, 0, MPI_COMM_WORLD);
	}
}

void relocationPoints(int n, double time, Point *arrPoints)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < n; i++)
	{
		arrPoints[i].x = arrPoints[i].x + time*arrPoints[i].vX;
		arrPoints[i].y = arrPoints[i].y + time*arrPoints[i].vY;
	}

}

//Find the best group of clusters.
//Updates clusters array.
void kMeansAlgorithm(int myid,int n, int k, int limit, Point *arrPoints, Cluster *arrClusters,Point *cudaPoints,MPI_Datatype MPI_POINT_TYPE, MPI_Datatype MPI_CLUSTER_TYPE)
{
	int i;
	boolean isPointChangeGroup = TRUE;
	boolean tempIsPointChangeGroup;
	// initial k first points as clusters center.
	initialClusters(myid,n, k, arrPoints, arrClusters,MPI_CLUSTER_TYPE);
	
	for (i = 0; i < limit && isPointChangeGroup; i++)
	{	
		// assume that point will not change the current group.
		isPointChangeGroup = FALSE; 
		tempIsPointChangeGroup =  ClassifyWithCudaPreparation(n,k,arrPoints,cudaPoints,arrClusters);
		
		//check if was a change of group - sum all the process tempIsPointChangeGroup result and distribute it back- 
		//-to isPointChangeGroup
		MPI_Allreduce(&tempIsPointChangeGroup, &isPointChangeGroup, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		
		//all processes check the isPointChangeGroup, if the isPointChangeGroup positive ,keeping kmeans ,else done.
		if (isPointChangeGroup)
			recalculateClusterCenters(n, k, arrPoints, arrClusters);
	}

}

//Preparation Cuda to Classify points to clusters center.
//Copy clusters to gpu to all processes
//Call cuda classify function
//Return True if was a change, else False - for each process.
boolean ClassifyWithCudaPreparation(int n, int k,Point *arrPoints,Point *cudaPoints, Cluster *arrClusters)
{
	cudaError_t cudaStatus;
	Cluster* cudaClusteres;
	
	cudaClusteres = copyClusteresToGPU(arrClusters, k);

	boolean isChange = classifyEachPointToClustereCentersWithCuda(n, k, cudaPoints, cudaClusteres);
	// Copy point vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(arrPoints, cudaPoints, n * sizeof(Point), cudaMemcpyDeviceToHost);
	checkCudaStatus(cudaStatus, "cudaMemcpy failed!");

	cudaFree(cudaClusteres);
	return isChange;
}

//Copy clusters array to gpu.
Cluster* copyClusteresToGPU(Cluster* arrClusters, int k) {
	Cluster* cudaClusteres;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&cudaClusteres, k * sizeof(Cluster));
	checkCudaStatus(cudaStatus, "cudaMalloc failed! allocate Clusteres");

	cudaStatus = cudaMemcpy(cudaClusteres, arrClusters, k * sizeof(Cluster), cudaMemcpyHostToDevice);
	checkCudaStatus(cudaStatus, "cudaMemcpy failed! Clusteres Copy");

	return cudaClusteres;
}

// Master Choose the first k point as a cluster center.
// Master brodcast the arrClusters to all processes.
void initialClusters(int myid,int n, int k, Point *arrPoints, Cluster *arrClusters,MPI_Datatype MPI_CLUSTER_TYPE)
{
	int i;
	if (myid == MASTER)
	{
	#pragma omp parallel for
		for (i = 0; i < k; i++)
		{
			arrClusters[i].x = arrPoints[i].x;
			arrClusters[i].y = arrPoints[i].y;
			arrClusters[i].id = i;
			arrClusters[i].diameter = 0;
			arrClusters[i].numOfGroupPoints = 0;
		}
	}
	//brodcast to slaves arrClusters.
	MPI_Bcast(arrClusters, k, MPI_CLUSTER_TYPE, 0, MPI_COMM_WORLD);
}

void resetClustersData(int k, Cluster *arrClusters)
{
	int i;
	//reset clusters data
	for (i = 0; i < k; i++)
	{
		arrClusters[i].numOfGroupPoints = 0;
		arrClusters[i].diameter = 0;
	}
}

//All processes Calculate sum of x,y , then synchronize with all proccess the result to generate new center.
void recalculateClusterCenters(int n, int k, Point *arrPoints, Cluster *arrClusters)
{
	int i, j;
	double currentSumX;
	double currentSumY;

	resetClustersData(k, arrClusters);

	// sum of all the clusters for each process.
#pragma omp parallel for
	for (i = 0; i < k; i++)
	{
		currentSumX = 0;
		currentSumY = 0;
		for (j = 0; j < n; j++)
		{
			if (arrPoints[j].clusterId == arrClusters[i].id) // in the current - i group
			{
				arrClusters[i].numOfGroupPoints++;
				currentSumX += arrPoints[j].x;
				currentSumY += arrPoints[j].y;
			}
		}
		arrClusters[i].x = currentSumX;
		arrClusters[i].y = currentSumY;
	}

	synchronizeClusterCenters(k,arrClusters);
}


void synchronizeClusterCenters(int k, Cluster *arrClusters)
{
	int i, totalPointInGroup ;
	double x, y;

	for (i = 0; i < k; i++)
	{
		//sum coord x and y in the cluster i from each process
		MPI_Allreduce(&(arrClusters[i].x), &x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&(arrClusters[i].y), &y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		// sum amount group of points in the cluster from each process 
		MPI_Allreduce(&(arrClusters[i].numOfGroupPoints), &totalPointInGroup, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		arrClusters[i].x = x / totalPointInGroup;
		arrClusters[i].y = y / totalPointInGroup;
	}

}

//Calcualte the qualtiy from current clusters and points 
void qualityEvaluate(int n, int k, Point *arrPoints, Cluster *arrClusters, double *quality)
{
	int i, j;
	double distance = 0;
	double tempQual = 0;

	calculateClustersDiameter(n, k, arrPoints, arrClusters);

#pragma omp parallel for private(distance,j) reduction(+ : tempQual)
	for (i = 0; i < k; i++)
	{
		for (j = 0; j < k; j++)
		{
			if (i != j)
			{
				distance = distanceClusterToCluster(arrClusters[i], arrClusters[j]);
				tempQual += (arrClusters[i].diameter / distance);
			}
		}
	}
	int average = k*(k - 1);
	tempQual /= average;
	*quality = tempQual;
}

// Calculate for each cluster diameter
// Updates diameter attribute for each cluster
void calculateClustersDiameter(int n, int k, Point *arrPoints, Cluster *arrClusters)
{
	int i;
	#pragma omp parallel for 
	for (i = 0; i < k; i++)
	{
		arrClusters[i].diameter = diameter(n, k, arrPoints, arrClusters[i].id);
	}
}

//Calcualte the diameter for given clusterId
//Return diameter
double diameter(int n, int k, Point *arrPoints, int clusterId)
{
	int i, j;
	double distance;
	double maxDist = 0;
	for (i = 0; i < n; i++)
	{
		if (arrPoints[i].clusterId == clusterId) //if point i in the group 
		{
			for (j = i + 1; j < n; j++) // check all the other point.
			{
				if (arrPoints[j].clusterId == clusterId)
				{
					distance = distancePointToPoint(arrPoints[i], arrPoints[j]);
					if (distance > maxDist)
						maxDist = distance;
				}
			}
		}
	}

	return maxDist;
}

double distanceClusterToCluster(Cluster c1, Cluster c2)
{
	double deltaX = pow(c2.x - c1.x, 2);
	double deltaY = pow(c2.y - c1.y, 2);
	return sqrt(deltaX + deltaY);
}

//Calculate distance from point to other point.
double distancePointToPoint(Point p1, Point p2)
{
	double deltaX = pow(p1.x - p2.x, 2);
	double deltaY = pow(p1.y - p2.y, 2);
	return sqrt(deltaX + deltaY);
}