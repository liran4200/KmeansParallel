#include "header.h"

int main(int argc, char *argv[])
{
	Point   *arrPoints   = NULL;
	Cluster *arrClusters = NULL;
	int myid, numprocs;
	int k = 0, n = 0, limit = 0;
	double dt=0, qm=0, intervalT = 0, bestQuality = 0, momentTime = 0,  executionTime;
	clock_t start, end;
	const char* fileNameOutput = "C:\\Users\\liran yehudar\\Documents\\Visual Studio 2015\\Projects\\Kmeans_Parallel_liran\\Kmeans_Parallel_liran\\output.txt";
	const char* fileNameInput  = "C:\\Users\\liran yehudar\\Documents\\Visual Studio 2015\\Projects\\Kmeans_Parallel_liran\\Kmeans_Parallel_liran\\data2.txt";

	// init all the MPI functionality
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// master 
	if (myid == MASTER)
	{
		arrPoints = readFromFile(&n, &k, &intervalT, &dt, &limit, &qm, fileNameInput);
		start = clock();
		printf("Master read file, num procs =%d \n",numprocs);
		fflush(stdout);
	}

	// all proccesses
	arrClusters = kMeansPreparationProcesses(numprocs, myid, n, k, intervalT, dt, limit, qm, arrPoints, &bestQuality, &momentTime);

	if (myid == MASTER)
	{
		end = clock();
		executionTime = (double)(end - start)/ CLOCKS_PER_SEC;
		writeToFile(k, arrClusters, momentTime, executionTime, bestQuality, fileNameOutput);
		printf("Done! Wrote to output file\n");
		fflush(stdout);
		free(arrPoints);
	}

	//all processes free clusters
	free(arrClusters);
	MPI_Finalize();
	return 0;
}

// read data from file
Point* readFromFile(int *n, int *k, double *time, double *dt, int *limit, double *qm, const char *fileName)
{
	Point *points = NULL;
	int i;
	FILE *fp;
	fopen_s(&fp, fileName, "r");

	if (fp == NULL) {
		printf("Couldnt open the file\n");
		exit(1);
	}

	fscanf_s(fp, "%d %d %lf %lf %d %lf\n", n, k, time, dt, limit, qm);

	points = (Point*)malloc(sizeof(Point)*(*n));
	if (points == NULL)
	{
		printf("points allocation error\n");
		exit(2);
	}
	for (i = 0; i < *n; i++) {
		double x, y, vx, vy;
		fscanf_s(fp, "%lf %lf %lf %lf\n", &x, &y, &vx, &vy);
		points[i].x = x;
		points[i].y = y;
		points[i].vX = vx;
		points[i].vY = vy;
	}

	fclose(fp);
	return points;
}
// write data to file
void writeToFile(int k, Cluster *arrCluster, double momentTime, double executionTime, double qualtiy, const char *filename)
{
	FILE *fp;
	int i;

	fopen_s(&fp, filename, "w");
	if (fp == NULL) {
		printf("Couldnt open the file\n");
		exit(1);
	}

	fprintf_s(fp, "First occurrence at t = %lf with q = %lf\nCenters of the clusters:\n", momentTime, qualtiy);
	for (i = 0; i < k; i++)
	{
		fprintf_s(fp, "Id: %d, ( %lf , %lf )\n", arrCluster[i].id, arrCluster[i].x, arrCluster[i].y);
	}

	fprintf_s(fp, "\nexecution time is: %f second\n", executionTime);
	fclose(fp);
}

