#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"
#include "MathVector.h"

using namespace std;

template <class T>
bool mXmSizeCheck(Matrix<T> m1, Matrix<T> m2)
{
    if (m1.getColSize() == m2.getRowSize()) return true;
    return false;
}

template <class T>
bool vPlusvSizeCheck(MathVector<T> v1, MathVector<T> v2)
{
    if ((v1.getSize()) == (v2.getSize())) return true;
    return false;
}

template <class T>
bool mPlusmSizeCheck(Matrix<T> m1, Matrix<T> m2)
{
    if (m1.getColSize() == m2.getColSize() && m1.getRowSize() == m2.getRowSize()) return true;
    return false;
}

template <class T>
bool hadamardXSizeCheck(Matrix<T> m1, Matrix<T> m2)
{
    return(mPlusmSizeCheck(m1, m2));
}

template <class T>
bool mPlusMtSizeCheck(Matrix<T> m1, Matrix<T> m2)
{
    if (m1.getColSize() == m2.getColSize()) return true;
    return false;
}

template <class T>
Matrix<T> vectorToMatrix(MathVector<T> v)
{
	vector<vector<T>> tmp;
	tmp.push_back(v.getVector());
	return *(new Matrix<T>(tmp, 1, v.getSize()));
}

template <class T>
MathVector<T> addVector(T a, MathVector<T> x, MathVector<T> y)
{
	return (x * a) + y;
}

template <class T>
MathVector<T> vectorDot(MathVector<T> v1, MathVector<T> v2)
{
	T result = 0;
	for (int i = 0; i < v1.getSize(); i++)
	{
		result += (v1->vectorData[i] * v2->vectorData[i]);
	}
	return result;
}

template <class T>
Matrix<T> hadamardX(Matrix<T> m1, Matrix<T> m2)
{
    vector<vector<T>> tmp;
    tmp.reserve(m1.getColSize());
    for (int i = 0; i < m1.getColSize(); ++i) {
        vector<T> tmpRow;
        tmpRow.reserve(m1.getRowSize());
        for (int j = 0; j < m1.getRowSize(); ++j) {
            tmpRow.push_back(m1.getMatrix()[i][j] * m2.getMatrix()[i][j]);
        }
        tmp.push_back(tmpRow);
    }

    Matrix<T> tmpMatrix(tmp, m1.getColSize(), m1.getRowSize());
    return tmpMatrix;
}

template <class T>
Matrix<T> vectorXMatrix(MathVector<T> v, Matrix<T> m)
{
    return matrixXmatrix(vectorToMatrix(v), m);
}

template <class T>
Matrix<T> matrixTXMatrix(Matrix<T> m1, Matrix<T> m2)
{
    vector<vector<T>> tmp;
    tmp.reserve(m1.getRowSize());
    for (int i = 0; i < m1.getRowSize(); i++)
    {
        vector<T> tmpCol;
        tmpCol.reserve(m2.getRowSize());
        for (int j = 0; j < m2.getRowSize(); j++)
        {
            T elem = 0;

            for (int k = 0; k < m1.getColSize(); k++)
            {
                elem += m1.getMatrix()[k][i] * m2.getMatrix()[k][j];
            }

            tmpCol.push_back(elem);
        }
        tmp.push_back(tmpCol);
    }
    Matrix<T> tmpMatrix(tmp, m1.getRowSize(), m2.getRowSize());
    return tmpMatrix;
}

template <class T>
void matrixXMatrixT(Matrix<T> &m1, Matrix<T> &m2, Matrix<T> &result)
{
    vector<vector<T>> tmp;
    tmp.reserve(m1.getColSize());
    for (int i = 0; i < m1.getColSize(); i++)
    {
        vector<T> tmpCol;
        tmpCol.reserve(m2.getColSize());
        for (int j = 0; j < m2.getColSize(); j++)
        {
            T elem = 0;

            for (int k = 0; k < m1.getRowSize(); k++)
            {
                elem += m1.getMatrix()[i][k] * m2.getMatrix()[j][k];
            }

            tmpCol.push_back(elem);
        }
        tmp.push_back(tmpCol);
    }
    result.set(tmp, m1.getColSize(), m2.getColSize());
}