#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <random>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bitset>

using namespace std;



void CheckErrors(cudaError_t status);
bool CompareResults(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

template<unsigned long long k>
class SequenceOfBits;
class CudaTimer;

//Number of bits per sequence
const unsigned long long sequence_length = 1000;

//Number of sequences
const unsigned long long input_sequence_count = 1000ull;
const unsigned long long comparisons = (((input_sequence_count*(input_sequence_count - 1)) / 2));

ostream & operator<<(ostream & out, SequenceOfBits<sequence_length> & sequence);
SequenceOfBits<sequence_length> * GenerateInput();
vector<pair<int, int> > ToPairVector(const SequenceOfBits<comparisons> & result_sequence);
void HammingCPU(SequenceOfBits<sequence_length> * sequence, SequenceOfBits<comparisons> * odata);
vector<pair<int, int> > FindPairsGPU(SequenceOfBits<sequence_length> * h_sequence);
vector<pair<int, int> > FindPairsCPU(SequenceOfBits<sequence_length> * sequence);
void PrintSequences(SequenceOfBits<sequence_length> * arr);




template<unsigned int N>
class DeviceResults;
class Results
{
public:
	unsigned int **result_array;
};

//Host holds the array
template<unsigned int N>
class HostResults : public Results
{
public:

	HostResults()
	{
		//N - 1 is equal to the number of rows. No need to create row with 0 fields
		result_array = new unsigned int* [N - 1];

		for (int i = 0; i < N - 1; i++)
		{
			//Results are from ballot, which returns 32bits. That is why we are counting how many full 32bit words we need for each row 
			int number_of_words = (int)(ceil((i + 1) / 32.0));			
			result_array[i] = new unsigned int[number_of_words];
		}
	}

	

	char GetBit(unsigned int row, unsigned int col) const
	{
		//We never use row 0 , as it is left down triangle matrix without main diagonal, so row 0 is empty. No need to create it then
		//We hold each column in one bit, so we need to get it by & with 1 moved to the correct spot
		return (char)(result_array[row - 1][col / 32] >> (col % 32) & 1);
	}

	void CopyRows(const DeviceResults<N> & array, unsigned int beginning, unsigned int count)
	{
		unsigned int **temp_array = new unsigned int*[count];

		
		cudaMemcpy(temp_array, array.result_array + beginning - 1, count * sizeof(unsigned int*), cudaMemcpyDeviceToHost);
		//calculate how many full 32bit words we need and create array with the size of it
		for (int i = 0; i < count; ++i)
		{
			int number_of_words = (int)(ceil((beginning + i) / 32.0));
			cudaMemcpyAsync(result_array[beginning - 1 + i], temp_array[i], sizeof(unsigned int) * number_of_words, cudaMemcpyDeviceToHost);
		}
			

		delete[] temp_array;
	}

	HostResults(HostResults<N> &&h_result)
	{
		this->result_array = h_result.result_array;
		h_result.result_array = nullptr;
	}

	HostResults<N>&& operator=(HostResults<N> &&h_result)
	{
		this->result_array = h_result.result_array;
		h_result.result_array = nullptr;
	}

	~HostResults()
	{
		if (result_array == nullptr)
			return;

		for (int i = 0; i < N - 1; i++)
		{
			delete[] result_array[i];
		}

		delete[] result_array;
	}
	
};

//similar like HostResults, but for device . Device copies their result to it after they finish checking
template<unsigned int N>
class DeviceResults : public Results
{
public:

	HostResults<N> ToHostArray()
	{
		HostResults<N> host;
		unsigned int * temporal[N - 1];
		CheckErrors(cudaMemcpy(temporal, result_array, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; ++i)
		{
			unsigned int number_of_words = (unsigned int)(ceil((i + 1) / 32.0));
			CheckErrors(cudaMemcpyAsync(host.result_array[i], temporal[i], sizeof(unsigned int) * number_of_words , cudaMemcpyDeviceToHost));
		}
		CheckErrors(cudaDeviceSynchronize());
		return host;
	}

	DeviceResults()
	{
		//N - 1 is equal to the number of rows. No need to create row with 0 fields
		CheckErrors(cudaMalloc(&result_array, sizeof(unsigned int* ) * (N - 1)));

		unsigned int* temporal[N - 1];
		for (int i = 0; i < N - 1; ++i)
		{
			double length = (ceil((i + 1) / 32.0));
			CheckErrors(cudaMalloc(&(temporal[i]), sizeof(unsigned int) * length));
			CheckErrors(cudaMemset(temporal[i], 0, sizeof(unsigned int) * length));
		}
		CheckErrors(cudaMemcpyAsync(result_array, &(temporal[0]), sizeof(unsigned int*)*(N - 1), cudaMemcpyHostToDevice));
		CheckErrors(cudaDeviceSynchronize());
	}

	

	~DeviceResults()
	{
		unsigned int *temp_arr[N - 1];
		CheckErrors(cudaMemcpy(temp_arr, result_array, sizeof(unsigned int*) * (N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; i++)
		{
			CheckErrors(cudaFree(temp_arr[i]));
		}
		CheckErrors(cudaFree(result_array));
	}
};

template<unsigned int N>
vector<pair<int, int> > GetResultPairs(const HostResults<N> & result_array);

__host__ __device__ char compareSequences(SequenceOfBits<sequence_length> * sequence1, SequenceOfBits<sequence_length> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);
__host__ __device__ unsigned int* GetPointer(unsigned int **array, unsigned int row, unsigned int col);

const unsigned long long threads_in_block = 1024;
const unsigned long long rows_per_call = 35;
const unsigned long long words64bits_in_sequence = ((sequence_length + 63) / 64);

int main()
{
	cudaError_t cudaStatus;
	//Need more shared memory than the default allocation
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	printf("Generation sequence in progress...");
	SequenceOfBits<sequence_length>* sequence = GenerateInput();
	printf("Completed!\n");
	
	printf("Starting searching for pairs of sequences with Hamming distance equal 1 on GPU...\n");
	auto resultsGPU = FindPairsGPU(sequence);
	printf("Completed!\n");

	printf("Starting searching for pairs of sequences with Hamming distance equal 1 on CPU...\n");
	auto resultsCPU = FindPairsCPU(sequence);
	printf("Completed!\n");

	printf("Comparing GPU results with CPU results...\n");
	CompareResults(resultsGPU, resultsCPU);
	//PrintSequences(sequence);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

template<unsigned long long N>
class SequenceOfBits
{
public:
	// !!(N%64) returns 0 if N is divisible by 64 and 1 if it is not. Array must contain whole 64-bits long words.
	static const unsigned long long array_size = (N / 64 + (!!(N % 64))) * 8;

	__host__ __device__ SequenceOfBits() {}

	__host__ __device__ SequenceOfBits(const SequenceOfBits<N> & sequence)
	{
		memcpy(array, sequence.array, array_size * 8);
	}

	__host__ __device__ const SequenceOfBits<N> & operator=(const SequenceOfBits<N> & sequence)
	{
		memcpy(array, sequence.array, array_size * 8);
		return sequence;
	}

	//Get a 32-bits long word, so 32/8 = 4 bytes long.
	__host__ __device__ inline unsigned int *Get32BitsWord(unsigned int word_index)
	{
		return (unsigned int*)(array + word_index * 32 / 8);
	}

	//Get a 64-bits long word , so 64/8 = 8 bytes long word
	__host__ __device__ inline unsigned long long *Get64BitsWord(unsigned long long word_index)
	{
		return (unsigned long long*)(array + word_index * 64 / 8);
	}

	//Char has 1 byte so 8 bits. Divide by 8 to get the byte with our searched bit,
	//move it to the right so our bit is the least significant bit of the byte.
	//Then we can & it with 1 to get the value.
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return array[index / 8] >> (index % 8) & 1;
	}

	//Can't write only one bit, have to write whole byte. Get the byte containing our bit , get a mask of it with ones everywhere and zero on the bit spot.
	//After &-ing the mask with the byte we get the same byte, but with 0 in place of our bit.
	//Afterwards we can & it with a byte containing zeroes everywhere and our desired value in the place of the searched bit to write the correct value in the desired bit spot.
	//!! makes 0 from 0 and 1 from non-zero value
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
	}		
private:
	char array[array_size];
};

__host__ __device__ char compareSequences(SequenceOfBits<sequence_length> * first_sequence, SequenceOfBits<sequence_length> * second_sequence)
{
	int difference_count = 0;
	//Words are 64bits, and (sequence_length + 63) / 64 works as ceil(sequence_length/64)
	for (int i = 0; i < (sequence_length + 63) / 64; ++i)
	{
		unsigned long long int first_word, second_word, xor_result;
		first_word = *(first_sequence->Get64BitsWord(i));
		second_word = *(second_sequence->Get64BitsWord(i));

		xor_result = first_word ^ second_word;
		//if xor_result & (xor_result - 1) == 0 that means that was not a power of 2, so we stop. Otherwise that means there was a difference on exactly one place.
		difference_count += xor_result == 0 ? 0 : (xor_result & (xor_result - 1) ? 2 : 1);

		if (difference_count > 1)		
			return 0;
		
	}
	//returns 0 if difference_count = 0 and 1 otherwise
	return !!difference_count;
}

//Function to get the (x, y) coordinates from a N x N matrix based on the index. We care only about what is above the main diagonal.
//On the main diagonal we have a comparison with themselfs (H(a, a) = 0 ), and below it we have duplicates, as H(a, b) == H(b, a).
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j)
{
	//adding 1 to k to skip first result
	*i = (unsigned int)ceil((0.5 * (-1 + sqrt(1 + 8 * (k + 1)))));
	//decreasing 1 from j , as we start from 0 not 1
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((unsigned long long)(*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

void HammingCPU(SequenceOfBits<sequence_length> * sequence, SequenceOfBits<comparisons> * odata)
{
	int x = 1, y = 0;
	for (unsigned long long k = 0; k < comparisons / 32; ++k)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; ++i)
		{
			//Setting the result one bit at the time. compareSequences returns 0 or 1, so we set the value in the correct place in the result.
			result |= (unsigned int)(compareSequences(sequence + x, sequence + y)) << i;
			++y;
			if (y == x)
			{
				++x;
				y = 0;
			}
		}
		*(odata->Get32BitsWord(k)) = result;
	}
	//Something left, not a whole word
	if (comparisons % 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < comparisons % 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + x, sequence + y)) << i;
			++y;
			if (y == x)
			{
				++x;
				y = 0;
			}
		}
		//on the missing places there will be zeroes
		*(odata->Get32BitsWord(comparisons / 32)) = result;
	}
}

class CudaTimer
{
public:
	CudaTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		started = false;
	}

	~CudaTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		started = true;
		cudaEventRecord(start);
		cudaEventSynchronize(start);
	}

	float Stop()
	{
		if (!started)
			return -1.0f;
		float ms;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&ms, start, stop);
		started = false;
		return ms;
	}
private:
	bool started;
	cudaEvent_t start, stop;
};


bool CompareResults(const vector<pair<int, int>>& gpu_result, const vector<pair<int, int> > & cpu_result)
{
	unsigned long long shorter_vector_length;
	unsigned long long cpu_pair_count = cpu_result.size();
	unsigned long long gpu_pair_count = gpu_result.size();

	if (gpu_pair_count < cpu_pair_count)
		shorter_vector_length = gpu_pair_count;
	else
		shorter_vector_length = cpu_pair_count;

	vector<pair<int, int>> result_gpu(gpu_result);
	vector<pair<int, int>> result_cpu(cpu_result);

	//sorting to make sure the pairs are in the same order, to be able to compare the.
	sort(result_cpu.begin(), result_cpu.end());
	sort(result_gpu.begin(), result_gpu.end());
	

	const vector<pair<int, int> > & longer_vector = cpu_pair_count > gpu_pair_count ? result_cpu : result_gpu;
	bool equal = true;
	
	if (gpu_pair_count != cpu_pair_count)
	{
		cout << "Number of elements in both results is not equal (GPU: " << gpu_pair_count << ", CPU: " << cpu_pair_count << ") !" << endl;
		equal = false;
	}
	else
	{
		cout << "Number of elements in both result is equal (GPU: " << gpu_pair_count << ", CPU: " << cpu_pair_count << ")" << endl;
	}

	//need to have access to the last number that was checked
	int i = 0;
	for (; i < shorter_vector_length; ++i)
	{
		if (result_gpu[i] != result_cpu[i])
		{
			equal = false;
			cout << "Difference on pair number " << i << "; GPU Pair: (" << result_gpu[i].first << ", " << result_gpu[i].second << ") CPU Pair: ("
				<< result_cpu[i].first << ", " << result_cpu[i].second << ")" << endl;			
		}		

	}
	if (cpu_pair_count != gpu_pair_count)
	{
		cout << "Remaining pairs from " << ((cpu_pair_count > gpu_pair_count) ? "CPU" : "GPU") << " result:" << endl;
		for (; i < longer_vector.size(); ++i)
		{
			cout << "(" << longer_vector[i].first << ", " << longer_vector[i].second << ")" << endl;
		}
	}

	if (equal)
		printf("Results are the same!\n");

	return equal;
}

//We never use row 0 , as it is left down triangle matrix, row 0 either contains comparison with themself (0, 0) or duplicates. No need to create it then.
__host__ __device__ unsigned int* GetPointer(unsigned int **array, unsigned int row, unsigned int col)
{
	return array[row - 1] + col / 32;
}

ostream & operator<<(ostream & out, SequenceOfBits<sequence_length> & sequence)
{
	for (unsigned long long i = 0; i < sequence_length; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}

SequenceOfBits<sequence_length> * GenerateInput()
{
	mt19937_64 source;
	int seed = random_device()();
	source.seed(seed);
	SequenceOfBits<sequence_length> * result = new SequenceOfBits<sequence_length>[input_sequence_count];

	memset(result, 0, sizeof(SequenceOfBits<sequence_length>) * input_sequence_count);
	
	for (int i = 0; i < input_sequence_count; i++)
	{
		//generate it 64 bits at the time
		for (int j = 0; j < SequenceOfBits<sequence_length>::array_size / 8 - 1; ++j)
		{
			*(result[i].Get64BitsWord(j)) = source();
		}
		//last word can be not full, so we generate in separately. We move the 64bit word so we are left with only the required number of set bits and with the rest of them set to 0
		*(result[i].Get64BitsWord(sequence_length / 64)) = source() >> (64 - (sequence_length % 64));
	}	

	return result;
}

vector<pair<int, int> > ToPairVector(const SequenceOfBits<comparisons> & result_sequence)
{
	vector<pair<int, int> > resultVector;
	for (unsigned long long k = 0; k < comparisons; k++)
	{
		if (result_sequence.GetBit(k))
		{
			unsigned int i, j;
			k2ij(k, &i, &j);
			resultVector.push_back(make_pair(i, j));
		}
	}

	return resultVector;
}


vector<pair<int, int> > FindPairsCPU(SequenceOfBits<sequence_length> * sequence)
{
	SequenceOfBits<comparisons> *odata;
	odata = new SequenceOfBits<comparisons>();
	CudaTimer timerCall;
	timerCall.Start();
	HammingCPU(sequence, odata);
	float result_time = timerCall.Stop();
	printf("CPU execution time: %f\n", result_time);
	vector<pair<int, int> > result = ToPairVector(*odata);
	delete odata;
	return result;
}






__global__ void HammingGPU(SequenceOfBits<sequence_length> *sequences, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seq_no = tid + column_offset;
	char res[rows_per_call];
	char m = rows_per_call > row_offset ? row_offset : rows_per_call;
	memset(res, 0, rows_per_call * sizeof(char));

	SequenceOfBits<sequence_length> & s = *(sequences + seq_no);
	__shared__ SequenceOfBits<sequence_length> ar[rows_per_call];
	for (unsigned int offset = 0; offset < m * words64bits_in_sequence; offset += blockDim.x)
	{
		unsigned int oid = threadIdx.x + offset;
		if (oid < m * words64bits_in_sequence)
		{
			*(ar[oid / words64bits_in_sequence].Get64BitsWord(oid % words64bits_in_sequence)) =
				*((sequences + row_offset - oid / words64bits_in_sequence)->Get64BitsWord(oid % words64bits_in_sequence));
			//printf("Thread %d wrote %llu\n", oid, *(ar[oid / words64bits_in_sequence].Get64BitsWord(oid % words64bits_in_sequence)));
		}
	}
	__syncthreads();
	for (int j = 0; j < words64bits_in_sequence; ++j)
	{
		unsigned long long sf = *(s.Get64BitsWord(j));
		for (int i = 0; i < m; ++i)
		{
			//unsigned int seq2_no = row_offset - i;
			if (res[i] <= 1)
			{
				unsigned long long s = *(ar[i].Get64BitsWord(j));

				unsigned long long xor_result = sf ^ s;
				char r = (xor_result ? ((xor_result & (xor_result - 1)) ? 2 : 1) : 0);
				//printf("%d %d %d %d % d\n", seq2_no, seq_no, j, i, r);
				res[i] += r;
			}
		}
	}
	for (int i = 0; i < m; ++i)
	{
		unsigned int b;
		unsigned int seq2_no = row_offset - i;
		char v = res[i] == 1;

		__syncthreads();
		b = __ballot(v);

		//printf("%d %d %d %d %d\n", seq_no, seq2_no, (int)v, b, (int)res[i]);
		if (seq2_no > seq_no && !(seq_no % 32))
		{
			*(GetPointer(arr, seq2_no, seq_no)) = b;
		}
	}
}

vector<pair<int, int> > FindPairsGPU(SequenceOfBits<sequence_length> * h_sequence)
{
	SequenceOfBits<sequence_length> *d_idata;
	DeviceResults<input_sequence_count> d_result;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	unsigned long long inputSize = sizeof(SequenceOfBits<sequence_length>)* input_sequence_count;
	timerMemory.Start();
	CheckErrors(cudaMalloc(&d_idata, inputSize));
	CheckErrors(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	timerCall.Start();
	for (int j = input_sequence_count - 1; j > 0; j -= rows_per_call)
	{
		if (j >= threads_in_block)
		{
			HammingGPU << < j / threads_in_block, threads_in_block >> > (d_idata, d_result.result_array, j, 0);
		}
		// Same as j % threads_in_block > 0
		if (j % threads_in_block)
		{
			HammingGPU << < 1, j % threads_in_block >> > (d_idata, d_result.result_array, j, j - (j%threads_in_block));
			
		}
		
	}
	CheckErrors(cudaDeviceSynchronize());
	xtime = timerCall.Stop();
	HostResults<input_sequence_count> h_result(d_result.ToHostArray());
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	printf("GPU Times : execution: %f, with memory: %f\n", xtime, xmtime);
	vector<pair<int, int> > res = GetResultPairs(h_result);
	return res;
}

template<unsigned int N>
vector<pair<int, int> > GetResultPairs(const HostResults<N> & result_array)
{
	vector<pair<int, int> > result;
	for (int i = 1; i < N; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			if (result_array.GetBit(i, j))
			{
				result.push_back(make_pair(i, j));
			}
		}
	}

	return result;
}



void PrintSequences(SequenceOfBits<sequence_length> * array)
{
	for (int i = 0; i < input_sequence_count; ++i)
	{
		cout << array[i] << endl;
		printf("\n");
	}
		
}

void CheckErrors(cudaError_t status)
{
	if (status != cudaSuccess)
		printf("Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));
}