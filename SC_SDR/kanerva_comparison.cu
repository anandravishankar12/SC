#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// HELPER FUNCTIONS
// Print an array of floats in [,,] format
void printFloatArray(float *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%.2f, ", arr[i]);
	}
	printf("%.2f]", arr[len-1]);

	printf("\n");	
}

// Print an array of unsigned ints in [,,] format
void printUnsignedArray(unsigned *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%u, ", arr[i]);
	}
	printf("%u]", arr[len-1]);

	printf("\n");	
}

// Print an array of ints in [,,] format
void printArray(int *arr, int len){
	printf("[");
	for (int i = 0; i < len -1; ++i) {
		printf("%d, ", arr[i]);
	}
	printf("%d]", arr[len-1]);

	printf("\n");	
}


// A simple helper function to compute the difference between to points in time
double time_diff(struct timeval x , struct timeval y){
	double x_ms , y_ms , diff;
	 
	x_ms = (double)x.tv_sec*1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec*1000000 + (double)y.tv_usec;
	 
	diff = (double)y_ms - (double)x_ms;
	 
	return diff;
}

// A simple helper function to reset two arrays with random values
void resetTestData(float *floatArr, int lenFloats, int *intArr, int lenInts){
	for (int i = 0; i < lenFloats; i++){
		floatArr[i] = (float)rand()/(float)(RAND_MAX/1.0);
	}
	for (int i = 0; i < lenInts; i++){
		intArr[i] = (int)rand()/(float)(RAND_MAX/10);
	}
}


// returns the tile indicies corresponding to the floats and ints
void getFeaturesNorm(float **prototypes, int numPrototypes, float *floats, int lenFloats, int *ints, int lenInts, int numCoordinates, float threshold, int *features) {
	
	for (int i = 0; i < numPrototypes; i++) {
		float *prototype = prototypes[i];
		float distance = 0.0;

		float diff = 0.0;

		// Compute using norm

		for (int j = 0; j < lenFloats; j++) {
			diff = floats[j] - prototype[j];
			distance += diff*diff;
		}

		for (int j = 0; j < lenInts; j++){
			diff = (float) ints[j] - prototype[j+lenFloats];
			distance += diff*diff;
		}

		if (sqrt(distance) < threshold){
			features[i] = 1;
		}
	}
}

// threadIdx.x = coord
// blockIdx.x = prototype
__global__ void calcFeatures(float *d_prototypes, float *d_floats, int lenFloats, int *d_ints, int lenInts, float *d_activationRadii, int *d_features){


	float val = 0.0;
	
	if (threadIdx.x < lenFloats){

		float distance = fabsf(d_floats[threadIdx.x] - d_prototypes[blockIdx.x * (lenFloats + lenInts) + threadIdx.x]);
		val = distance <= d_activationRadii[threadIdx.x] ? 1 - distance/d_activationRadii[threadIdx.x] : 0;
	} else {
		float distance = fabsf(((float) d_ints[threadIdx.x - lenFloats]) - d_prototypes[blockIdx.x * (lenFloats + lenInts) + threadIdx.x]);
		val = distance <= d_activationRadii[threadIdx.x] ? 1 - distance/d_activationRadii[threadIdx.x] : 0;
	}

	atomicAnd(&d_features[blockIdx.x], val > 0 ? 1 : 0);

}

// TODO finish this
void parallel_getFeaturesActivationRadii(int numPrototypes, int numCoordinates, float *d_prototypes, float *h_floatArr, float *d_floats, int lenFloats, int *h_intArr, int *d_ints, int lenInts, float *d_activationRadii, int *d_features, int *h_features){

	cudaMemset(d_features, 0xF, numPrototypes*sizeof(int)); 

	cudaMemcpy(d_floats, h_floatArr, lenFloats*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ints, h_intArr, lenInts * sizeof(int), cudaMemcpyHostToDevice);

	calcFeatures<<<numPrototypes, numCoordinates>>>(d_prototypes, d_floats, lenFloats, d_ints, lenInts, d_activationRadii, d_features);

	cudaMemcpy(h_features, d_features, numPrototypes * sizeof(float), cudaMemcpyDeviceToHost);

}


// returns the tile indicies corresponding to the floats and ints
void getFeaturesActivationRadii(int numPrototypes, int numCoordinates, float *prototypes,float *floats, int lenFloats, int *ints, int lenInts, float *activationRadii, int *features) {


	for (int i = 0; i < numPrototypes; i++) {

		float minValue = INFINITY;
		float distance;
		float val;

		// Do floats
		for (int j = 0; j < lenFloats; j++) {

			distance = fabs(floats[j] - prototypes[i*lenFloats + j]);

			val = distance <= activationRadii[j] ? 1 - distance/activationRadii[j] : 0;


			minValue = minValue < val ? minValue : val;
		}

		// Do ints
		for (int j = 0; j < lenInts; j++) {
			distance = fabs((float)ints[j] - prototypes[i*lenFloats + j]);

			val = distance <= activationRadii[j + lenFloats] ? 1 - distance/activationRadii[j + lenFloats] : 0;


			minValue = minValue < val ? minValue : val;
		}

		// if close enough, activate feature
		features[i] = minValue > 0 ? 1 : 0;
	}
}

int main(int argc, char ** argv) {

	// Use random other than 1
	// srand ( time(NULL) );

	// not testing ints so set it to length 0
	int h_intArr[0] = {};
	int lenInts = 0;

	int * d_ints;
	cudaMalloc((void **) &d_ints, lenInts*sizeof(int));


	int maxPrototypes = 2048;

	int maxFloats = 1024;

	int numTrials = 5000;
	int incrementBy = 10;

	struct timeval beforeA, afterA, beforeB, afterB;

	double sumA = 0.0, avgTimeA = 0.0, minTimeA = INFINITY, maxTimeA = 0.0, sumB = 0.0, avgTimeB = 0.0, minTimeB = INFINITY, maxTimeB = 0.0;
	int maxTimeTrialA = 0, maxTimeTrialB = 0, minTimeTrialA = 0, minTimeTrialB;

	for (int numPrototypes = 200; numPrototypes < maxPrototypes; numPrototypes*=2){


		int features[numPrototypes];
		int testFeatures[numPrototypes];

		int *d_features;
		cudaMalloc((void **) &d_features, numPrototypes*sizeof(int));

		for (int lenFloats = 2; lenFloats < maxFloats; lenFloats+=incrementBy){


			int numCoordinates = lenFloats + lenInts;

			float h_prototypes[numPrototypes*numCoordinates];
			// initialize random prototypes
			resetTestData(h_prototypes, numPrototypes*lenFloats, h_intArr, lenInts);


			float *d_prototypes;
			cudaMalloc((void **) &d_prototypes, numPrototypes*numCoordinates*sizeof(float));
			cudaMemcpy(d_prototypes, h_prototypes, numPrototypes * numCoordinates * sizeof(float), cudaMemcpyHostToDevice);


			// populate the activation radii array, although .2 could be passed in, there could be different radii for different dimensions
			float h_activationRadii[lenFloats];
			for (int i = 0; i < lenFloats; i++){
				h_activationRadii[i] = .2;
			}

			float *d_activationRadii;
			cudaMalloc((void **) &d_activationRadii, lenFloats * sizeof(float));
			cudaMemcpy(d_activationRadii, h_activationRadii, lenFloats * sizeof(float), cudaMemcpyHostToDevice);

			float h_floatArr[lenFloats];

			float *d_floats;
			cudaMalloc((void **) &d_floats, lenFloats * sizeof(float));


			for (int trial = 0; trial < numTrials; trial++){

				// reset float array
				resetTestData(h_floatArr, lenFloats, h_intArr, lenInts);


				// time the Parallel tiles
				gettimeofday(&beforeA , NULL);
				parallel_getFeaturesActivationRadii(numPrototypes, numCoordinates, d_prototypes, h_floatArr, d_floats, lenFloats, h_intArr, d_ints, lenInts, d_activationRadii, d_features, testFeatures);
				gettimeofday(&afterA , NULL);


				// time the Serial tiles
				gettimeofday(&beforeB, NULL);
				getFeaturesActivationRadii(numPrototypes, numCoordinates, h_prototypes, h_floatArr, lenFloats, h_intArr, lenInts, h_activationRadii, features);
				gettimeofday(&afterB, NULL);


				// confirm correct calculation
				int Errors = 0;
				for (int j = 0; j < numPrototypes; j++){
					if (features[j] != testFeatures[j]){
						printf("Error: Incorrect Arrays\nCorrect Array:  ");
						printArray(features, numPrototypes);
						printf("\nComputed Array: ");
						printArray(testFeatures, numPrototypes);
						Errors = 1;
						break;
					}
				}
				if (Errors){
					// if there is an error (differing arrays), free the memory and print debug info 
					cudaFree(d_floats);
					cudaFree(d_prototypes);
					cudaFree(d_activationRadii);
					cudaFree(d_ints);
					cudaFree(d_features);
					printf("Error: numPrototypes %d, lenFloats %d, trial %d\n", numPrototypes, lenFloats, trial);
					return 1;
				}


				// compute time comparison
				double timeTakenA = time_diff(beforeA , afterA);
				sumA += timeTakenA;

				if (timeTakenA < minTimeA){
					minTimeA = timeTakenA;
					minTimeTrialA = trial;
				}
				if (timeTakenA > maxTimeA){
					maxTimeA = timeTakenA;
					maxTimeTrialA = trial;
				}

				//compute time comparison
				double timeTakenB = time_diff(beforeB , afterB);
				sumB += timeTakenB;

				if (timeTakenB < minTimeB){
					minTimeB = timeTakenB;
					minTimeTrialB = trial;
				}
				if (timeTakenB > maxTimeB){
					maxTimeB = timeTakenB;
					maxTimeTrialB = trial;
				}

			} // trialsloop

			cudaFree(d_floats);
			cudaFree(d_prototypes);
			cudaFree(d_activationRadii);

			// compute the average time for each scenario
			avgTimeA= sumA/numTrials;
			avgTimeB = sumB/numTrials;

			if (avgTimeA < avgTimeB){
				printf("numPrototypes: %d\t numCoordinates: %d\n", numPrototypes, numCoordinates);
				printf("\tParallel\n\t\tMin Time: %.0lf us | Min Trial: %d | Max Time: %.0lf us | Max Trial: %d | Avg time : %.0lf us\n\tSerial\n\t\tMin Time: %.0lf us | Min Trial: %d | Max Time: %.0lf us | Max Trial: %d | Avg time : %.0lf us\n\n", minTimeA, minTimeTrialA, maxTimeA, maxTimeTrialA, avgTimeA, minTimeB, minTimeTrialB, maxTimeB, maxTimeTrialB, avgTimeB); 
				printf("---------------------------------------------------------\n");
				break;
			}
		} // float loop

		cudaFree(features);
		cudaFree(testFeatures);
		cudaFree(d_features);

	} // prototype loop


	return 0;
}