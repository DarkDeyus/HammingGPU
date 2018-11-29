#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

#define BITS_IN_SEQUENCE 1000 //Number of bits in one sequence
#define INPUT_SEQUENCE_SIZE 1000ull //Number of sequences
#define COMPARISONS (((INPUT_SEQUENCE_SIZE*(INPUT_SEQUENCE_SIZE - 1)) / 2)) //Number of comparisons
#define THREADS_PER_BLOCK 1024 //Number of threads run in one block
#define SEQUENCES_PER_CALL 35 //Number of rows taken once per function call

void CheckErrors(cudaError_t status);

template<unsigned long long k>
class SequenceOfBits;
class CudaTimer;

template<unsigned int N>
class DeviceResults;

class Results
{
public:
	unsigned int **arr;
};

template<unsigned int N>
class HostResults : public Results
{
public:

	HostResults()
	{
		arr = new unsigned int* [N - 1];

		for (int i = 0; i < N - 1; i++)
		{
			arr[i] = new unsigned int[(int)(ceil((i + 1) / 32.0))];
		}
	}

	~HostResults()
	{
		if (arr == nullptr)
			return;

		for (int i = 0; i < N - 1; i++)
		{
			delete[] arr[i];
		}

		delete[] arr;
	}

	HostResults<N>&& operator=(HostResults<N> &&h_result)
	{
		this->arr = h_result.arr;
		h_result.arr = nullptr;
	}

	HostResults(HostResults<N> &&h_result)
	{
		this->arr = h_result.arr;
		h_result.arr = nullptr;
	}

	void CopyRows(const DeviceResults<N> & array, unsigned int start, unsigned int quantity)
	{
		unsigned int **temp_arr = new unsigned int*[quantity];
		cudaMemcpy(temp_arr, array.arr + start - 1, quantity * sizeof(unsigned int*), cudaMemcpyDeviceToHost);
		for (int i = 0; i < quantity; ++i)
		{
			cudaMemcpyAsync(arr[start - 1 + i], temp_arr[i], sizeof(unsigned int) * (int)(ceil((start + i) / 32.0)), cudaMemcpyDeviceToHost);
		}
		delete[] temp_arr;
	}

	char GetBit(unsigned int row, unsigned int col) const
	{
		return (char)(arr[row - 1][col / 32] >> (col % 32) & 1);
	}
};

template<unsigned int N>
class DeviceResults : public Results
{
public:
	DeviceResults()
	{
		CheckErrors(cudaMalloc(&arr, sizeof(unsigned int*)*(N - 1)));
		unsigned int* temp_arr[N - 1];
		for (int i = 0; i < N - 1; ++i)
		{
			CheckErrors(cudaMalloc(&(temp_arr[i]), sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
			CheckErrors(cudaMemset(temp_arr[i], 0, sizeof(unsigned int) * (ceil((i + 1) / 32.0))));
		}
		CheckErrors(cudaMemcpyAsync(arr, &(temp_arr[0]), sizeof(unsigned int*)*(N - 1), cudaMemcpyHostToDevice));
		CheckErrors(cudaDeviceSynchronize());
	}

	~DeviceResults()
	{
		unsigned int *temp_arr[N - 1];
		CheckErrors(cudaMemcpy(temp_arr, arr, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; i++)
		{
			CheckErrors(cudaFree(temp_arr[i]));
		}
		CheckErrors(cudaFree(arr));
	}

	HostResults<N> ToHostArray()
	{
		HostResults<N> host;
		unsigned int * temp_arr[N - 1];
		CheckErrors(cudaMemcpy(temp_arr, arr, sizeof(unsigned int*)*(N - 1), cudaMemcpyDeviceToHost));
		for (int i = 0; i < N - 1; ++i)
		{
			CheckErrors(cudaMemcpyAsync(host.arr[i], temp_arr[i], sizeof(unsigned int) * (unsigned int)(ceil((i + 1) / 32.0)), cudaMemcpyDeviceToHost));
		}
		CheckErrors(cudaDeviceSynchronize());
		return host;
	}
};


__host__ __device__ char compareSequences(SequenceOfBits<BITS_IN_SEQUENCE> * sequence1, SequenceOfBits<BITS_IN_SEQUENCE> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);
void HammingCPU(SequenceOfBits<BITS_IN_SEQUENCE> * sequence, SequenceOfBits<COMPARISONS> * odata);
bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

ostream & operator<<(ostream & out, SequenceOfBits<BITS_IN_SEQUENCE> & sequence);
SequenceOfBits<BITS_IN_SEQUENCE> * GenerateInput();
vector<pair<int, int> > ToPairVector(const SequenceOfBits<COMPARISONS> & result_sequence);
void PrintAsMatrix(const SequenceOfBits<COMPARISONS> & sequence, ostream & stream);

vector<pair<int, int> > FindPairsGPU(SequenceOfBits<BITS_IN_SEQUENCE> * h_sequence);
vector<pair<int, int> > FindPairsCPU(SequenceOfBits<BITS_IN_SEQUENCE> * sequence);
__host__ __device__ unsigned int* GetPointer(unsigned int **array, unsigned int row, unsigned int col);

template<unsigned int N>
vector<pair<int, int> > ToPairVector(const HostResults<N> & result_array);

void PrintArray(SequenceOfBits<BITS_IN_SEQUENCE> * arr);

int main()
{
	cudaError_t cudaStatus;
	//Need more shared memory than the default allocation
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	printf("Generation sequence in progress...");
	SequenceOfBits<BITS_IN_SEQUENCE>* sequence = GenerateInput();
	printf("Completed!\n");
	
	printf("Starting searching for pairs of sequences with Hamming distance equal 1 on GPU...\n");
	auto resultsGPU = FindPairsGPU(sequence);
	printf("Completed!\n");

	printf("Starting searching for pairs of sequences with Hamming distance equal 1 on CPU...\n");
	auto resultsCPU = FindPairsCPU(sequence);
	printf("Completed!\n");

	printf("Comparing GPU results with CPU results...\n");
	ComparePairs(resultsGPU, resultsCPU);

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

__host__ __device__ char compareSequences(SequenceOfBits<BITS_IN_SEQUENCE> * first_sequence, SequenceOfBits<BITS_IN_SEQUENCE> * second_sequence)
{
	int difference_count = 0;
	//Words are 64bits, and (BITS_IN_SEQUENCE + 63) / 64 works as ceil(BITS_IN_SEQUENCE/64)
	for (int i = 0; i < (BITS_IN_SEQUENCE + 63) / 64; ++i)
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

void HammingCPU(SequenceOfBits<BITS_IN_SEQUENCE> * sequence, SequenceOfBits<COMPARISONS> * odata)
{
	int x = 1, y = 0;
	for (unsigned long long k = 0; k < COMPARISONS / 32; ++k)
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
	if (COMPARISONS % 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < COMPARISONS % 32; i++)
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
		*(odata->Get32BitsWord(COMPARISONS / 32)) = result;
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


bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result)
{
	unsigned long long gsize = gpu_result.size(), csize = cpu_result.size();
	unsigned long long n = gsize < csize ? gsize : csize;

	vector<pair<int, int> > gpu_res(gpu_result);
	vector<pair<int, int> > cpu_res(cpu_result);
	sort(gpu_res.begin(), gpu_res.end());
	sort(cpu_res.begin(), cpu_res.end());

	const vector<pair<int, int> > & gv = csize > gsize ? cpu_res : gpu_res;
	bool equal = true;

	if (gsize != csize)
	{
		cout << "Number of elements is not equal (GPU: " << gsize << ", CPU: " << csize << ") !" << endl;
		equal = false;
	}
	else
	{
		cout << "Number of elements are equal (GPU: " << gsize << ", CPU: " << csize << ")" << endl;
	}

	int i;
	for (i = 0; i < n; ++i)
	{
		if (gpu_res[i] != cpu_res[i])
		{
			cout << "Difference on " << i << ": GPU: (" << gpu_res[i].first << ", " << gpu_res[i].second << ") CPU: ("
				<< cpu_res[i].first << ", " << cpu_res[i].second << ")" << endl;
			equal = false;
		}
		else
		{
			//cout << "Pair " << i << ": GPU: (" << gpu_res[i].first << ", " << gpu_res[i].second << ") CPU: ("
			//		<< cpu_res[i].first << ", " << cpu_res[i].second << ")" << endl;

		}

	}
	if (csize != gsize)
	{
		cout << "Rest pairs on " << ((csize > gsize) ? "CPU" : "GPU") << ":" << endl;
		for (; i < gv.size(); ++i)
		{
			cout << "(" << gv[i].first << ", " << gv[i].second << ")" << endl;
		}
	}
	if (equal)
	{
		cout << "Results are the same" << endl;
	}
	return equal;
}



ostream & operator<<(ostream & out, SequenceOfBits<BITS_IN_SEQUENCE> & sequence)
{
	for (unsigned long long i = 0; i < BITS_IN_SEQUENCE; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}

SequenceOfBits<BITS_IN_SEQUENCE> * GenerateInput()
{
	srand(2018);

	SequenceOfBits<BITS_IN_SEQUENCE> * r = new SequenceOfBits<BITS_IN_SEQUENCE>[INPUT_SEQUENCE_SIZE];

	memset(r, 0, sizeof(SequenceOfBits<BITS_IN_SEQUENCE>)*INPUT_SEQUENCE_SIZE);

	for (int i = 0; i < INPUT_SEQUENCE_SIZE; i++)
	{
		*(r[i].Get32BitsWord(0)) = i;
	}
	return r;
}

vector<pair<int, int> > ToPairVector(const SequenceOfBits<COMPARISONS> & result_sequence)
{
	vector<pair<int, int> > result;
	for (unsigned long long k = 0; k < COMPARISONS; k++)
	{
		if (result_sequence.GetBit(k))
		{
			unsigned int i, j;
			k2ij(k, &i, &j);
			result.push_back(make_pair(i, j));
		}
	}

	return result;
}

void PrintAsMatrix(const SequenceOfBits<COMPARISONS> & sequence, ostream & stream)
{
	for (int i = 0; i < INPUT_SEQUENCE_SIZE; ++i)
	{
		for (int j = 0; j < INPUT_SEQUENCE_SIZE; ++j)
		{
			if (j <= i)
			{
				cout << "  ";
			}
			else
			{
				cout << (short int)sequence.GetBit(ij2k(i, j)) << " ";
			}
		}
		cout << endl;
	}
}

vector<pair<int, int> > FindPairsCPU(SequenceOfBits<BITS_IN_SEQUENCE> * sequence)
{
	SequenceOfBits<COMPARISONS> *odata;
	odata = new SequenceOfBits<COMPARISONS>();
	CudaTimer timerCall;
	timerCall.Start();
	HammingCPU(sequence, odata);
	float result_time = timerCall.Stop();
	printf("CPU execution time: %f\n", result_time);
	vector<pair<int, int> > result = ToPairVector(*odata);
	delete odata;
	return result;
}



#define WORDS64_IN_SEQUENCE ((BITS_IN_SEQUENCE + 63) / 64)


__global__ void HammingGPU(SequenceOfBits<BITS_IN_SEQUENCE> *sequences, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seq_no = tid + column_offset;
	char res[SEQUENCES_PER_CALL];
	char m = SEQUENCES_PER_CALL > row_offset ? row_offset : SEQUENCES_PER_CALL;
	memset(res, 0, SEQUENCES_PER_CALL * sizeof(char));

	SequenceOfBits<BITS_IN_SEQUENCE> & s = *(sequences + seq_no);
	__shared__ SequenceOfBits<BITS_IN_SEQUENCE> ar[SEQUENCES_PER_CALL];
	for (unsigned int offset = 0; offset < m * WORDS64_IN_SEQUENCE; offset += blockDim.x)
	{
		unsigned int oid = threadIdx.x + offset;
		if (oid < m * WORDS64_IN_SEQUENCE)
		{
			*(ar[oid / WORDS64_IN_SEQUENCE].Get64BitsWord(oid % WORDS64_IN_SEQUENCE)) =
				*((sequences + row_offset - oid / WORDS64_IN_SEQUENCE)->Get64BitsWord(oid % WORDS64_IN_SEQUENCE));
			//printf("Thread %d wrote %llu\n", oid, *(ar[oid / WORDS64_IN_SEQUENCE].Get64BitsWord(oid % WORDS64_IN_SEQUENCE)));
		}
	}
	__syncthreads();
	for (int j = 0; j < WORDS64_IN_SEQUENCE; ++j)
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

vector<pair<int, int> > FindPairsGPU(SequenceOfBits<BITS_IN_SEQUENCE> * h_sequence)
{
	SequenceOfBits<BITS_IN_SEQUENCE> *d_idata;
	DeviceResults<INPUT_SEQUENCE_SIZE> d_result;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	unsigned long long inputSize = sizeof(SequenceOfBits<BITS_IN_SEQUENCE>)* INPUT_SEQUENCE_SIZE;
	timerMemory.Start();
	CheckErrors(cudaMalloc(&d_idata, inputSize));
	CheckErrors(cudaMemcpy(d_idata, h_sequence, inputSize, cudaMemcpyHostToDevice));
	timerCall.Start();
	for (int j = INPUT_SEQUENCE_SIZE - 1; j > 0; j -= SEQUENCES_PER_CALL)
	{
		if (j >= THREADS_PER_BLOCK)
		{
			HammingGPU << < j / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_idata, d_result.arr, j, 0);			
		}
		// Same as j % THREADS_PER_BLOCK > 0
		if (j % THREADS_PER_BLOCK)
		{
			HammingGPU << < 1, j % THREADS_PER_BLOCK >> > (d_idata, d_result.arr, j, j - (j%THREADS_PER_BLOCK));
			
		}
		
	}
	CheckErrors(cudaDeviceSynchronize());
	xtime = timerCall.Stop();
	HostResults<INPUT_SEQUENCE_SIZE> h_result(d_result.ToHostArray());
	xmtime = timerMemory.Stop();
	cudaFree(d_idata);
	printf("GPU Times : execution: %f, with memory: %f\n", xtime, xmtime);
	vector<pair<int, int> > res = ToPairVector(h_result);
	return res;
}

template<unsigned int N>
vector<pair<int, int> > ToPairVector(const HostResults<N> & result_array)
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

__host__ __device__ unsigned int* GetPointer(unsigned int **array, unsigned int row, unsigned int col)
{
	return array[row - 1] + col / 32;
}

void PrintArray(SequenceOfBits<BITS_IN_SEQUENCE> * array)
{
	for (int i = 0; i < INPUT_SEQUENCE_SIZE; ++i)	
		cout << array[i] << endl;	
}

void CheckErrors(cudaError_t status)
{
	if (status != cudaSuccess)
		printf("Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));
}