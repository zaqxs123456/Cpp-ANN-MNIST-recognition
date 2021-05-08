#include <iostream>
#include "NeuralNetwork.cpp"
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>

using namespace std;

template <class T>
void dataLoader(vector<Matrix<T>> &, vector<Matrix<T>> &, string);

int main() {

    NeuralNetwork<float> neuralNetwork;
    vector<Matrix<float>> X_train;
    vector<Matrix<float>> Y_train;
    vector<Matrix<float>> X_test;
    vector<Matrix<float>> Y_test;
    bool loaded = false;

    while(true)
    {
        cout << "new: Create new neural network. \t " << endl;
        cout << "load: Load neural network. \t " << endl;
        if(loaded)
        {
            cout << "train: Train the neural network. \t " << endl;
            cout << "predict: Predict a random data from test data. \t " << endl;
            cout << "predictAll: Predict all data from test data. \t " << endl;
            cout << "predictToWrong: Predict data from test data until find a wrong predict. \t " << endl;
        }
        cout << "exit: Exit. \t " << endl;

        string input;
        cin >> input;

        if(input == "new")
        {
            string dataPath, trainPath, hiddenLayerSizeString;
            vector<int> hiddenLayerSize;
            cout << "Input these for default: " << endl;
            cout << "../data/train.txt ../data/test.txt 30" << endl;
            cout << endl;

            cout << "Training Data Path: (e.g. ../data/train.txt)" << endl;
            cin >> dataPath;

            cout << "Test Data Path: (e.g. ../data/test.txt)" << endl;
            cin >> trainPath;

            cout << "Hidden layer sizes: (e.g. 30)" << endl;
            cin >> hiddenLayerSizeString;
            {
                int size;
                char comma;
                stringstream ss(hiddenLayerSizeString);
                while (ss >> size) {
                    ss >> comma;
                    hiddenLayerSize.push_back(size);
                }
            }
            dataLoader(X_train, Y_train, dataPath);
            dataLoader(X_test, Y_test, trainPath);
            neuralNetwork.set(X_train, Y_train, X_test, Y_test, hiddenLayerSize);
            loaded = true;
            cout << "Created!" << endl;
            cout << endl;
        }
        else if(input == "load")
        {
            string dataPath, trainPath, loadPath;
            vector<int> hiddenLayerSize;
            cout << "Input these for default: " << endl;
            cout << "../data/train.txt ../data/test.txt save/train.ann" << endl;
            cout << endl;

            cout << "Training Data Path: (e.g. ../data/train.txt)" << endl;
            cin >> dataPath;

            cout << "Test Data Path: (e.g. ../data/test.txt)" << endl;
            cin >> trainPath;

            cout << "Weights and biases Path: (e.g. save/train.ann)" << endl;
            cin >> loadPath;

            dataLoader(X_train, Y_train, dataPath);
            dataLoader(X_test, Y_test, trainPath);
            neuralNetwork.set(loadPath, X_train, Y_train, X_test, Y_test);
            loaded = true;
            cout << "Loaded!" << endl;
            cout << endl;
        }
        else if(input == "train" && loaded)
        {
            int batchSize = 128, epoch = 10000;
            float learningRate = 1;
            string saveLocation, saveName;
            cout << "Input these for default: " << endl;
            cout << "20 128 1 save train" << endl;
            cout << endl;


            cout << "Epoch: (e.g. 20)" << endl;
            cin >> epoch;

            cout << "Batch size: (e.g. 128)" << endl;
            cin >> batchSize;

            cout << "Learning rate: (e.g. 1)" << endl;
            cin >> learningRate;

            cout << "Save location: (e.g. save)" << endl;
            cin >> saveLocation;

            cout << "Save name: (e.g. train)" << endl;
            cin >> saveName;

            neuralNetwork.train(epoch, batchSize, learningRate, saveLocation, saveName);
            cout << endl;
        }
        else if(input == "predict" && loaded)
        {
            neuralNetwork.samplePredict();
            cout << endl;
        }
        else if(input == "predictAll" && loaded)
        {
            neuralNetwork.predictAll();
            cout << endl;
        }
        else if(input == "predictToWrong" && loaded)
        {
            neuralNetwork.predictToWrong();
            cout << endl;
        }
        else if(input == "exit")
        {
            cout << endl;
            return 0;
        }
        else
        {
        cout << "invalid input." << endl;
        cout << endl;
        }
    }

}


template <class T>
void dataLoader(vector<Matrix<T>> &X_train, vector<Matrix<T>> &Y_train, string filePath)
{
    ifstream myFile(filePath);

    if (myFile.is_open())
    {
        cout << "Loading data ...\n";
        string line;
        vector<T> Y_trainWithNum;

        while (getline(myFile, line))
        {
            int x, y;
            stringstream ss(line);
            ss >> y;
            Y_trainWithNum.push_back(y);

            vector<vector<T>> tmpXCol;
            for (int i = 0; i < 28 * 28; i++) {
                ss >> x;
                vector<T> tmpXRow;
                tmpXRow.push_back(x/255.0);
                tmpXCol.push_back(tmpXRow);
            }
            Matrix<T> tmpX(tmpXCol, tmpXCol.size(),1);
            X_train.push_back(tmpX);
        }

        for (int k = 0; k < Y_trainWithNum.size(); ++k)
        {
            vector<vector<T>> tmpYCol;
            for (int j = 0; j < 10; j++)
            {
                vector<T> tmpYRow;
                tmpYRow.push_back((j == Y_trainWithNum[k])? 1:0);
                tmpYCol.push_back(tmpYRow);
            }
            Matrix<T> tmpY(tmpYCol, tmpYCol.size(),1);
            Y_train.push_back(tmpY);
        }


        myFile.close();


        cout << "Loading data finished.\n";
    }
    else
        cout << "Unable to open file" << '\n';

    return;
}