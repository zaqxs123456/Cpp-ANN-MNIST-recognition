#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

template <class T>
class Matrix
{
private:

	vector<vector<T>> matrixData;
	int size[2] = { 0,0 };

public:
    const static int thread_count = 32;

    Matrix(){}

    Matrix(vector<vector<T>> scalers)
    {
        set(scalers, scalers.size(), scalers[0].size());
    }

    Matrix(vector<vector<T>> scalers, int col, int row)
	{
		set(scalers, col, row);
	}

    Matrix(float from, float to, int col, int row)
    {
        setRandom(from, to, col, row);
    }

    Matrix(int col, int row)
    {
        setZero(col, row);
    }

	void set(vector<vector<T>> scalers, int col, int row)
	{
        size[0] = col;
        size[1] = row;
		matrixData = scalers;
	}

    void setRandom(float from, float to, int col, int row)
    {
        size[0] = col;
        size[1] = row;
        vector<vector<T>> scalers = fillRandom(from, to);
        matrixData.assign(scalers.begin(), scalers.end());
    }

    void setZero(int col, int row)
    {
        size[0] = col;
        size[1] = row;
        vector<vector<T>> scalers = fillZero();
        matrixData.assign(scalers.begin(), scalers.end());
    }


	Matrix<T> transpose()
	{
		vector<vector<T>> tmp;
        tmp.reserve(this->getRowSize());
		for (int i = 0; i < this->getRowSize(); i++)
		{
            vector<T> tmpRow;
            tmpRow.reserve(this->getColSize());
			for (int j = 0; j < this->getColSize(); j++)
			{
                tmpRow.push_back(this->getMatrix()[j][i]);
			}
            tmp.push_back(tmpRow);
		}
		Matrix<T> tmpMatrix(tmp, this->getRowSize(), this->getColSize());
		return tmpMatrix;
	}

	Matrix<T> operator * (Matrix<T> m)
	{
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());

        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(m.getRowSize());

            for (int j = 0; j < m.getRowSize(); j++)
            {
                T elem = 0;
                for (int k = 0; k < this->getRowSize(); k++)
                {
                    elem += this->getMatrix()[i][k] * m.getMatrix()[k][j];
                }
                tmpRow.push_back(elem);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), m.getRowSize());
        return tmpMatrix;
    }

    Matrix operator * (T constant)
    {
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(this->getRowSize());
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(this->getMatrix()[i][j] * constant);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    Matrix operator / (T constant)
    {
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(this->getRowSize());
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(this->getMatrix()[i][j] / constant);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    Matrix operator + (Matrix m)
    {
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(this->getRowSize());
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(this->getMatrix()[i][j] + m.getMatrix()[i][j]);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    Matrix  operator - (Matrix m)
    {
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(this->getRowSize());
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(this->getMatrix()[i][j]- m.getMatrix()[i][j]);
            }
            tmp.push_back(tmpRow);
        }
        Matrix<T> tmpMatrix(tmp, this->getColSize(), this->getRowSize());
        return tmpMatrix;
    }

    vector<vector<T>> fillRandom(float from, float to)
    {
        default_random_engine randEngine(time(NULL));
        uniform_real_distribution<T> realDist(from, to);
        realDist(randEngine);

        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.reserve(this->getRowSize());
            for (int j = 0; j < this->getRowSize(); j++)
            {
                tmpRow.push_back(realDist(randEngine));
            }
            tmp.push_back(tmpRow);
        }

        return tmp;
    }

    vector<vector<T>> fillZero()
    {
        vector<vector<T>> tmp;
        tmp.reserve(this->getColSize());
        for (int i = 0; i < this->getColSize(); i++)
        {
            vector<T> tmpRow;
            tmpRow.resize(this->getRowSize());
            tmp.push_back(tmpRow);
        }

        return tmp;
    }

    int getColSize() const
    {
        return size[0];
    }

    int getRowSize() const
    {
        return size[1];
    }

	vector<vector<T>>& getMatrix()
    {
		return matrixData;
	}

	void print()
	{
        for (int i = 0; i < this->getColSize(); i++)
        {
            for (int j = 0; j < this->getRowSize(); j++)
            {
                printf("%10.4f", matrixData[i][j]);
            }
            cout << endl;
        }
	}

	string toString()
    {
        stringstream ss;
        for (int i = 0; i < this->getColSize(); i++)
        {
            for (int j = 0; j < this->getRowSize(); j++)
            {
                ss << matrixData[i][j] << " ";
            }
            ss << endl;
        }
        return ss.str();
    }


};


