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
class BitSequence;
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


__host__ __device__ char compareSequences(BitSequence<BITS_IN_SEQUENCE> * sequence1, BitSequence<BITS_IN_SEQUENCE> * sequence2);
__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j);
__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j);
void Hamming1CPU(BitSequence<BITS_IN_SEQUENCE> * sequence, BitSequence<COMPARISONS> * odata);
void PrintComparison(const BitSequence<BITS_IN_SEQUENCE> & gpu_sequence, const BitSequence<BITS_IN_SEQUENCE> & cpu_sequence);
bool ComparePairs(const vector<pair<int, int> > & gpu_result, const vector<pair<int, int> > & cpu_result);

ostream & operator<<(ostream & out, BitSequence<BITS_IN_SEQUENCE> & sequence);
BitSequence<BITS_IN_SEQUENCE> * GenerateInput();
vector<pair<int, int> > ToPairVector(const BitSequence<COMPARISONS> & result_sequence);
void PrintAsMatrix(const BitSequence<COMPARISONS> & sequence, ostream & stream);

vector<pair<int, int> > FindPairsGPU(BitSequence<BITS_IN_SEQUENCE> * h_sequence);
vector<pair<int, int> > FindPairsCPU(BitSequence<BITS_IN_SEQUENCE> * sequence);
__host__ __device__ unsigned int* GetPointer(unsigned int **array, unsigned int row, unsigned int col);

template<unsigned int N>
vector<pair<int, int> > ToPairVector(const HostResults<N> & result_array);

void PrintArray(BitSequence<BITS_IN_SEQUENCE> * arr);

int main()
{
	cudaError_t cudaStatus;
	//Need more shared memory than the default allocation
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	printf("Generation sequence in progress...");
	BitSequence<BITS_IN_SEQUENCE>* sequence = GenerateInput();
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

template<unsigned long long k>
class BitSequence
{
public:
	__host__ __device__ BitSequence() {}

	//Char has 1 byte so 8 bits. Divide by 8 to get the byte with our searched bit,
	//move it to the right so our bit is the least significant bit of the byte.
	//Then we can & it with 1 to get the value.
	__host__ __device__ inline char GetBit(unsigned long long index) const
	{
		return array[index / 8] >> (index % 8) & 1;
	}

	//Can't write only one bit, have to write whole byte. Get the byte containing our bit , get a mask of it with ones everywhere and zero on the bit spot. After &-ing the mask with the byte we get the same byte, but with 0 in place of our bit.
	//Afterwards we can & it with a byte containing zeroes everywhere and our desired value in the place of the searched bit to write the correct value in the desired bit spot.
	//!! makes 0 from 0 and 1 from non-zero
	__host__ __device__ inline void SetBit(unsigned long long index, char value)
	{
		array[index / 8] = (array[index / 8] & (~(1 << (index % 8)))) | ((!!value) << (index % 8));
	}
	
	//Get a 32-bits long word, so 32/8 = 4 bytes long.
	__host__ __device__ inline unsigned int *GetWord32(unsigned int word_index)
	{
		return (unsigned int*)(array + word_index * 32 / 8);
	}

	//Get a 64-bits long word , so 64/8 = 8 bytes long word
	__host__ __device__ inline unsigned long long *GetWord64(unsigned long long word_index)
	{
		return (unsigned long long*)(array + word_index * 64 / 8);
	}

	__host__ __device__ BitSequence(const BitSequence<k> & sequence)
	{
		memcpy(array, sequence.array, arSize * 8);
	}

	__host__ __device__ const BitSequence<k> & operator=(const BitSequence<k> & sequence)
	{
		memcpy(array, sequence.array, arSize * 8);
		return sequence;
	}
	static const unsigned long long arSize = (k / 64 + (!!(k % 64))) * 8;

private:
	char array[arSize];
};

__host__ __device__ char compareSequences(BitSequence<BITS_IN_SEQUENCE> * sequence1, BitSequence<BITS_IN_SEQUENCE> * sequence2)
{
	int diff = 0;
	for (int j = 0; j < (BITS_IN_SEQUENCE + 63) / 64; ++j)
	{
		unsigned long long int a, b, axorb;
		a = *(sequence1->GetWord64(j));
		b = *(sequence2->GetWord64(j));
		axorb = a ^ b;
		diff += axorb == 0 ? 0 : (axorb & (axorb - 1) ? 2 : 1);
		if (diff > 1)
		{
			return 0;
		}
	}
	return !!diff;
}

__host__ __device__ void k2ij(unsigned long long k, unsigned int * i, unsigned int  * j)
{
	*i = (unsigned int)ceil((0.5 * (-1 + sqrt(1 + 8 * (k + 1)))));
	*j = (unsigned int)((k + 1) - 0.5 * (*i) * ((unsigned long long)(*i) - 1)) - 1;
}

__host__ __device__ unsigned long long ij2k(unsigned int i, unsigned int j)
{
	return ((unsigned long long)i) * (i - 1) / 2 + j;
}

void Hamming1CPU(BitSequence<BITS_IN_SEQUENCE> * sequence, BitSequence<COMPARISONS> * odata)
{
	unsigned long long numberOfComparisons = COMPARISONS;
	int i1 = 1, i2 = 0;
	for (unsigned long long k = 0; k < numberOfComparisons / 32; ++k)
	{
		unsigned int result = 0;
		for (int i = 0; i < 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + i1, sequence + i2)) << i;
			++i2;
			if (i2 == i1)
			{
				++i1;
				i2 = 0;
			}
		}
		*(odata->GetWord32(k)) = result;
	}
	if (numberOfComparisons % 32)
	{
		unsigned int result = 0;
		for (int i = 0; i < numberOfComparisons % 32; i++)
		{
			result |= (unsigned int)(compareSequences(sequence + i1, sequence + i2)) << i;
			++i2;
			if (i2 == i1)
			{
				++i1;
				i2 = 0;
			}
		}
		*(odata->GetWord32(numberOfComparisons / 32)) = result;
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

void PrintComparison(const BitSequence<BITS_IN_SEQUENCE> & gpu_sequence, const BitSequence<BITS_IN_SEQUENCE> & cpu_sequence)
{
	for (unsigned long long i = 0; i < INPUT_SEQUENCE_SIZE*(INPUT_SEQUENCE_SIZE - 1) / 2; ++i)
	{
		if (cpu_sequence.GetBit(i) != gpu_sequence.GetBit(i))
		{
			unsigned int i1, i2;
			k2ij(i, &i1, &i2);
			cout << "Difference on comparison number " << i << " (" << i1 << ", " << i2 << ") GPU " << (short int)gpu_sequence.GetBit(i) << " CPU " << (short int)cpu_sequence.GetBit(i) << endl;
		}
	}
}

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



ostream & operator<<(ostream & out, BitSequence<BITS_IN_SEQUENCE> & sequence)
{
	for (unsigned long long i = 0; i < BITS_IN_SEQUENCE; ++i)
	{
		out << (short int)sequence.GetBit(i);
	}
	return out;
}

BitSequence<BITS_IN_SEQUENCE> * GenerateInput()
{
	srand(2018);

	BitSequence<BITS_IN_SEQUENCE> * r = new BitSequence<BITS_IN_SEQUENCE>[INPUT_SEQUENCE_SIZE];

	memset(r, 0, sizeof(BitSequence<BITS_IN_SEQUENCE>)*INPUT_SEQUENCE_SIZE);

	for (int i = 0; i < INPUT_SEQUENCE_SIZE; i++)
	{
		*(r[i].GetWord32(0)) = i;
		/*for (int j = 0; j < BITS_IN_SEQUENCE / 32; j++)
		{
			*(r[i].GetWord32(j)) = rand() + rand()*RAND_MAX;
		}
		if(BITS_IN_SEQUENCE % 32)
			*(r[i].GetWord32(BITS_IN_SEQUENCE / 32)) = (rand() + rand()*RAND_MAX)%(1<<(BITS_IN_SEQUENCE%32));*/
	}
	return r;
}

vector<pair<int, int> > ToPairVector(const BitSequence<COMPARISONS> & result_sequence)
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

void PrintAsMatrix(const BitSequence<COMPARISONS> & sequence, ostream & stream)
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

vector<pair<int, int> > FindPairsCPU(BitSequence<BITS_IN_SEQUENCE> * sequence)
{
	BitSequence<COMPARISONS> *odata;
	odata = new BitSequence<COMPARISONS>();
	CudaTimer timerCall;
	timerCall.Start();
	Hamming1CPU(sequence, odata);
	float result_time = timerCall.Stop();
	printf("CPU execution time: %f\n", result_time);
	vector<pair<int, int> > result = ToPairVector(*odata);
	delete odata;
	return result;
}

__host__ __device__ inline char CompareWords64(const unsigned long long & first_word, const unsigned long long & second_word)
{
	unsigned long long xor_result = first_word ^ second_word;

	return (xor_result ? ((xor_result & (xor_result - 1)) ? 2 : 1) : 0);
}

#define WORDS64_IN_SEQUENCE ((BITS_IN_SEQUENCE + 63) / 64)


__global__ void HammingGPU(BitSequence<BITS_IN_SEQUENCE> *sequences, unsigned int **arr, unsigned int row_offset, unsigned int column_offset)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int seq_no = tid + column_offset;
	char res[SEQUENCES_PER_CALL];
	char m = SEQUENCES_PER_CALL > row_offset ? row_offset : SEQUENCES_PER_CALL;
	memset(res, 0, SEQUENCES_PER_CALL * sizeof(char));

	BitSequence<BITS_IN_SEQUENCE> & s = *(sequences + seq_no);
	__shared__ BitSequence<BITS_IN_SEQUENCE> ar[SEQUENCES_PER_CALL];
	for (unsigned int offset = 0; offset < m * WORDS64_IN_SEQUENCE; offset += blockDim.x)
	{
		unsigned int oid = threadIdx.x + offset;
		if (oid < m * WORDS64_IN_SEQUENCE)
		{
			*(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)) =
				*((sequences + row_offset - oid / WORDS64_IN_SEQUENCE)->GetWord64(oid % WORDS64_IN_SEQUENCE));
			//printf("Thread %d wrote %llu\n", oid, *(ar[oid / WORDS64_IN_SEQUENCE].GetWord64(oid % WORDS64_IN_SEQUENCE)));
		}
	}
	__syncthreads();
	for (int j = 0; j < WORDS64_IN_SEQUENCE; ++j)
	{
		unsigned long long sf = *(s.GetWord64(j));
		for (int i = 0; i < m; ++i)
		{
			//unsigned int seq2_no = row_offset - i;
			if (res[i] <= 1)
			{
				unsigned long long s = *(ar[i].GetWord64(j));
				char r = CompareWords64(sf, s);
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

vector<pair<int, int> > FindPairsGPU(BitSequence<BITS_IN_SEQUENCE> * h_sequence)
{
	BitSequence<BITS_IN_SEQUENCE> *d_idata;
	DeviceResults<INPUT_SEQUENCE_SIZE> d_result;
	CudaTimer timerCall, timerMemory;
	float xtime, xmtime;
	unsigned long long inputSize = sizeof(BitSequence<BITS_IN_SEQUENCE>)* INPUT_SEQUENCE_SIZE;
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

void PrintArray(BitSequence<BITS_IN_SEQUENCE> * array)
{
	for (int i = 0; i < INPUT_SEQUENCE_SIZE; ++i)	
		cout << array[i] << endl;	
}

void CheckErrors(cudaError_t status)
{
	if (status != cudaSuccess)
		printf("Cuda Error in %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(status));
}