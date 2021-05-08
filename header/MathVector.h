#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

template <class T>
class MathVector
{

private:

	vector<T> vectorData;
	int size = 0;
	float mean = 0.0f;
	float l1Norm = 0.0f;
	float l2Norm = 0.0f;

	void calMean()
	{
		for (int i = 0; i < size; i++)
		{
			mean += vectorData[i];
		}
		mean = mean / size;
	}

	void calL1Norm()
	{
		l1Norm = 0.0f;
		for (int i = 0; i < size; i++)
		{
			l1Norm += abs(vectorData[i]);
		}
	}

	void calL2Norm()
	{
		l2Norm = 0.0f;
		for (int i = 0; i < size; i++)
		{
			l2Norm += abs(vectorData[i] * vectorData[i]);
		}
		//l2Norm = this->sqrt(l2Norm);
        l2Norm = sqrtf(l2Norm);
	}

public:

	MathVector(T scalers[], int _size)
	{
		set(scalers, _size);
	}

	MathVector(vector<T> scalers, int _size)
	{
		set(scalers, _size);
	}

	void set(T scalers[], int _size)
	{
		size = _size;
		for (int i = 0; i < size; i++)
		{
			vectorData.push_back(scalers[i]);
		}

		calMean();
		calL1Norm();
		calL2Norm();
	}

	void set(vector<T> scalers, int _size)
	{
		size = _size;
		for (int i = 0; i < size; i++)
		{
			vectorData.push_back(scalers[i]);
		}

		calMean();
		calL1Norm();
		calL2Norm();
	}

	MathVector<T> operator + (const MathVector<T> v)
	{
        vector<T> _vector;
        for (int i = 0; i < size; i++)
        {
            _vector.push_back(this->vectorData[i] + v.vectorData[i]);
        }
        return MathVector(_vector, this->size);
	}

	MathVector operator * (const T scale)
	{
		vector<T> _vector;
		for (int i = 0; i < size; i++)
		{
			_vector.push_back(this->vectorData[i] * scale);
		}
		return MathVector(_vector, this->size);
	}

	void print()
	{
		for (int i = 0; i < size; i++)
		{
			cout << vectorData[i] << "\t";
		}
		cout << endl;
	}

	vector<T> getVector()
	{
		return vectorData;
	}

	int getSize()
	{
		return size;
	}

	float getMean()
	{
		return mean;
	}

	float getL1Norm()
	{
		return l1Norm;
	}

	float getL2Norm()
	{
		return l2Norm;
	}
};

