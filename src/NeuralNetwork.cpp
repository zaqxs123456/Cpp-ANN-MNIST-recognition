//
// Created by badaeib on 2019年11月14日.
//

#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <list>
#include <omp.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include "../header/MatrixVector.h"

using namespace std;
template <class T>
class NeuralNetwork
{
private:
    vector<int> layersNums; //num of neuron for each layer, include input layer.
    vector<Matrix<T>> weights;
    vector<Matrix<T>> biases;

    vector<Matrix<T>> inputLayers;
    vector<Matrix<T>> groundTruths;

    vector<Matrix<T>> testInputs;
    vector<Matrix<T>> testGroundTruths;
    //T (*activationFunction) ( T );

    int thread_count = 32;

    void iniWeightsAndBiases()
    {
        //loop through all layers.
        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            Matrix<T> weight(-1, 1, layersNums[i + 1], layersNums[i]);
            Matrix<T> bias(-1, 1, layersNums[i + 1], 1);
            weights.push_back(weight);
            biases.push_back(bias);
        }
    }

    Matrix<T> layerOutput(int layerIndex, Matrix<T> &preLayer)
    {
        Matrix actives = activation(weights[layerIndex] * preLayer + biases[layerIndex]);
        return actives;
    }

    void layerError(Matrix<T> &postWeights, Matrix<T> &postError, Matrix<T> &output, Matrix<T> &error)
    {
        error = hadamardX(matrixTXMatrix(postWeights, postError), activationD(output));
    }

    static T sigmoid (T x)
    {
        return 1.0/(1.0+ pow(M_E,-x));
    }
    static T sigmoidD (T sig)
    {
        return sig * (1.0-sig);
    }

    Matrix<T> activation(Matrix<T> layer)
    {
        vector<vector<T>> activesTmp;
        for (int i = 0; i < layer.getColSize(); i++)
        {
            vector<T> activeTmp;
            activeTmp.push_back(sigmoid(layer.getMatrix()[i][0]));
            activesTmp.push_back(activeTmp);
        }

        Matrix<T> actives(activesTmp, layer.getColSize(), 1);
        return actives;
    }

    Matrix<T> activationD(Matrix<T> &layer)
    {
        vector<vector<T>> activesTmp;
        for (int i = 0; i < layer.getColSize(); i++)
        {
            vector<T> activeTmp;
            activeTmp.push_back(sigmoidD(layer.getMatrix()[i][0]));
            activesTmp.push_back(activeTmp);
        }

        Matrix<T> actives(activesTmp, layer.getColSize(), 1);
        return actives;
    }

    void lastError(Matrix<T> &output, Matrix<T> &groundTruth, Matrix<T>&error)
    {
        error = hadamardX((output - groundTruth), activationD(output));
    }

    void outputs(Matrix<T> &inputLayer, vector<Matrix<T>> &outputs)
    {
        Matrix<T> output = inputLayer;
        outputs.push_back(output);

        for (int i = 0; i < layersNums.size() - 1; i++)
        {
            output = layerOutput(i, output);
            outputs.push_back(output);
        }
    }

    T loss()
    {
        T loss = 0;
        int total = testGroundTruths.size();
        int corr = total;

        #pragma omp parallel for default(none) shared(total) reduction( +:loss) reduction( -: corr)
        for (int i = 0; i < total; i++)
        {
            T singleLoss = 0;
            vector<Matrix<T>> _outputs;
            outputs(testInputs[i], _outputs);

            //if(!gTCompare(_outputs[_outputs.size() - 1], testGroundTruths[i])) corr--;
            if(predictByOutput(_outputs[_outputs.size() - 1]) != gTMatrixToNumber(testGroundTruths[i])) corr--;

            #pragma omp parallel for default(none) shared(_outputs, i) reduction( +:singleLoss)
            for (int j = 0; j < testGroundTruths[i].getColSize(); ++j)
            {
                singleLoss += pow(testGroundTruths[i].getMatrix()[j][0] - _outputs[_outputs.size() - 1].getMatrix()[j][0], 2.0f);
            }
            loss += singleLoss;
        }

        cout << "correctness: " << corr << "/" << total << " " <<  (float)(corr * 100.0f)/(float)total << "%" << endl;

        loss /= testInputs.size();
        loss /= 2;
        return loss;
    }

    bool gTCompare(Matrix<T> &lastLayers, Matrix<T> &groundTruth)
    {
        for (int i = 0; i < lastLayers.getColSize(); ++i)
        {
            if(round(lastLayers.getMatrix()[i][0]) != groundTruth.getMatrix()[i][0])
            {
                return  false;
            }
        }
        return true;
    }

    int predict(Matrix<T> &inputLayer)
    {
        vector<Matrix<T>> output;
        return predict(inputLayer, output);
    }

    int predictByOutput(Matrix<T> &output)
    {
        T highest = 0.0;
        int prediction = 0;
        for (int i = 0; i < output.getColSize(); ++i)
        {
            if(output.getMatrix()[i][0] >= highest)
            {
                highest = output.getMatrix()[i][0];
                prediction = i;
            }
        }
        return prediction;
    }

    int predict(Matrix<T> &inputLayer, vector<Matrix<T>> &output)
    {
        outputs(inputLayer, output);
        T highest = 0.0;
        int prediction = 0;
        for (int i = 0; i < output[output.size() - 1].getColSize(); ++i)
        {
            if(output[output.size() - 1].getMatrix()[i][0] >= highest)
            {
                highest = output[output.size() - 1].getMatrix()[i][0];
                prediction = i;
            }
        }
        return prediction;
    }

    int gTMatrixToNumber(Matrix<T> &gT)
    {
        int num = 0;
        for (int i = 0; i < 10; ++i)
        {
            if(gT.getMatrix()[i][0] == 1)
            {
                num = i;
            }
        }
        return num;
    }

    void update(vector<Matrix<T>> &batchInputLayer, vector<Matrix<T>> &batchGroundTruths, T learningRate)
    {
        clock_t start = clock();

        vector<Matrix<T>> batchWeightGradient;
        vector<Matrix<T>> batchBiasGradient;

        for (int k = 0; k < layersNums.size() - 1; ++k)
        {
            Matrix<T> emptyWeight(weights[k].getColSize(),weights[k].getRowSize());
            Matrix<T> emptyBias(biases[k].getColSize(),biases[k].getRowSize());

            batchWeightGradient.push_back(emptyWeight);
            batchBiasGradient.push_back(emptyBias);

        }

        #	pragma omp parallel for num_threads(thread_count) default(none) shared(batchInputLayer, batchGroundTruths, batchWeightGradient, batchBiasGradient)
        for (int i = 0; i < batchInputLayer.size(); ++i)
        {
            //cout << "i: " << i << endl;
            vector<Matrix<T>> _outputs;
            outputs(batchInputLayer[i], _outputs);

            vector<Matrix<T>> _errors;
            errors(_outputs, batchGroundTruths[i], _errors);

            #	pragma omp parallel for num_threads(thread_count) default(none) shared(_errors, _outputs, batchInputLayer, batchGroundTruths, batchWeightGradient, batchBiasGradient)
            for (int j = 0; j < layersNums.size() - 1; ++j)
            {
                Matrix<T> tmpGradient;
                matrixXMatrixT(_errors[j], _outputs[j], tmpGradient);

                #   pragma omp critical
                {
                    batchWeightGradient[j] = batchWeightGradient[j] + tmpGradient;
                    batchBiasGradient[j] = batchBiasGradient[j] + _errors[j];
                }
            }
        }

        for (int l = 0; l < layersNums.size() - 1; ++l)
        {
            weights[l] = weights[l] - batchWeightGradient[l] * learningRate /  batchInputLayer.size();
            biases[l] = biases[l] - batchBiasGradient[l] * learningRate /  batchInputLayer.size();
        }

        cout << "update take: " << (clock() - start) / (double) CLOCKS_PER_SEC << " sec." << endl;

    }

    void errors(vector<Matrix<T>> &outputs, Matrix<T> &groundTruth, vector<Matrix<T>>& _errors)
    {
        _errors.reserve(layersNums.size() - 1);

        list<Matrix<T>> errorsList;

        Matrix<T> errorTmp;
        lastError(outputs[outputs.size()-1], groundTruth, errorTmp);
        errorsList.push_front(errorTmp);

        for (int i = layersNums.size() - 2; i > 0; --i)
        {
            layerError(weights[i], errorTmp, outputs[i], errorTmp);
            errorsList.push_front(errorTmp);
        }
        for (Matrix<T> const &e: errorsList) {
            _errors.push_back(e);
        }
    }

    void toSaveForm(vector<Matrix<T>> &vMs, string &_output)
    {
        stringstream ss;
        ss << vMs.size() << endl << endl;
        for (int i = 0; i < vMs.size(); ++i)
        {
            ss << vMs[i].toString() << endl;
        }
        _output.append(ss.str());
    }

    void toSaveForm(vector<int> &vInts, string &_output)
    {
        stringstream ss;
        for (int vInt : vInts)
        {
            ss << to_string(vInt) << endl;
        }
        ss << endl;
        _output.append(ss.str());
    }

public:


    NeuralNetwork()= default;
    //inputLayers take column Matrix.
    NeuralNetwork(vector<Matrix<T>> inputLayers, vector<Matrix<T>> &groundTruths, vector<Matrix<T>> &testInputs, vector<Matrix<T>> &testGroundTruths, vector<int> &hiddenLayersNums)
    {
        set(inputLayers, groundTruths, testInputs, testGroundTruths, hiddenLayersNums);
    }

    NeuralNetwork(const string& location, vector<Matrix<T>> &inputLayers, vector<Matrix<T>> &groundTruths, vector<Matrix<T>> &testInputs, vector<Matrix<T>> &testGroundTruths)
    {
        set(location, inputLayers, groundTruths, testInputs, testGroundTruths);
    }


    void setTest(vector<Matrix<T>> &inputLayers, vector<Matrix<T>> &groundTruths, vector<Matrix<T>> &testInputs, vector<Matrix<T>> &testGroundTruths)
    {
        if(inputLayers.size() != groundTruths.size())
        {
            cout << "number of inputs: " << inputLayers.size() << endl;
            cout << "number of ground truths: " << groundTruths.size() << endl;
            throw invalid_argument( "number of inputs and ground truths not match!" );
        }

        if(testInputs.size() != testGroundTruths.size())
        {
            cout << "number of test inputs: " << testInputs.size() << endl;
            cout << "number of test ground truths: " << testGroundTruths.size() << endl;
            throw invalid_argument( "number of test inputs and ground truths not match!" );
        }
    }

    void set(vector<Matrix<T>> &setInputLayers, vector<Matrix<T>> &setGroundTruths, vector<Matrix<T>> &setTestInputs, vector<Matrix<T>> &setTestGroundTruths, vector<int> &hiddenLayersNums)
    {
        setTest(setInputLayers, setGroundTruths, setTestInputs, setTestGroundTruths);

        layersNums.push_back(setInputLayers[0].getColSize());
        layersNums.insert(this->layersNums.end(), hiddenLayersNums.begin(), hiddenLayersNums.end());
        layersNums.push_back(setGroundTruths[0].getColSize());
        iniWeightsAndBiases();
        inputLayers = setInputLayers;
        groundTruths =  setGroundTruths;

        testInputs = setTestInputs;
        testGroundTruths = setTestGroundTruths;
    }

    void set(const string& location, vector<Matrix<T>> &setInputLayers, vector<Matrix<T>> &setGroundTruths, vector<Matrix<T>> &setTestInputs, vector<Matrix<T>> &setTestGroundTruths)
    {
        setTest(setInputLayers, setGroundTruths, setTestInputs, setTestGroundTruths);

        load(location);

        inputLayers = setInputLayers;
        groundTruths = setGroundTruths;

        testInputs = setTestInputs;
        testGroundTruths = setTestGroundTruths;
    }

    void samplePredict()
    {
        default_random_engine randEngine(time(NULL));
        uniform_int_distribution<int> intDist(0, testInputs.size() - 1);
        intDist(randEngine);

        int rand = intDist(randEngine);

        vector<Matrix<T>> output;
        int resultPredicted = predict(testInputs[rand], output);
        int groundTruth = gTMatrixToNumber(testGroundTruths[rand]);

        cout << "Predicting " << rand <<  " data in testing pool." << endl;
        cout << "Predicted result: " << resultPredicted << endl;
        cout << "Ground truth: " << groundTruth << endl;
        if(resultPredicted == groundTruth)
        {
            cout << "The prediction is correct." << endl;
        }
        else
        {
            cout << "The prediction is wrong." << endl;
        }
        cout << "Prediction matrix:" << endl;
        output[output.size() - 1].transpose().print();
        cout << "Ground truth matrix:" << endl;
        testGroundTruths[rand].transpose().print();
    }

    void predictAll()
    {
        int correct = testInputs.size();
        for (int i = 0; i < testInputs.size(); ++i)
        {
            if(predict(testInputs[i]) != gTMatrixToNumber(testGroundTruths[i])) correct--;
        }
        cout << "correct: " << correct << " / " << testInputs.size() << endl;
        cout << "Correct percentage: " << (correct/(float)testInputs.size()) * 100 << "%" << endl;
    }

    void predictToWrong()
    {
        vector<int> index;
        for (int i = 0; i < testInputs.size(); ++i)
        {
            index.push_back(i);
        }

        default_random_engine randEngine(time(NULL));
        shuffle(begin(index), end(index), randEngine);

        int wrongIndex = -1;
        int resultPredicted = -1;
        int groundTruth = -1;

        vector<Matrix<T>> output;
        for (int i = 0; i < testInputs.size(); ++i)
        {
            resultPredicted = predict(testInputs[i], output);
            groundTruth = gTMatrixToNumber(testGroundTruths[i]);
            if(resultPredicted != groundTruth)
            {
                wrongIndex = i;
                break;
            }
        }
        if(wrongIndex == -1)
        {
            cout << "All correct!" << endl;
        }
        else
        {
            cout << "First wrong prediction: " << wrongIndex << endl;

            cout << "Predicted result: " << resultPredicted << endl;
            cout << "Ground truth: " << groundTruth << endl;

            cout << "Prediction matrix:" << endl;
            output[output.size() - 1].transpose().print();
            cout << "Ground truth matrix:" << endl;
            testGroundTruths[wrongIndex].transpose().print();
        }

}

    void printLayersNums()
    {
        for (int num: layersNums)
        {
            cout << num << endl;
        }
    }

    void printWeights()
    {
        int i = 0;
        for (Matrix<T> m: weights)
        {
            m.print();
        }
    }

    void printBiases()
    {
        for (Matrix<T> m: biases)
        {
            m.print();
        }
    }

    void train(int totalEpoch , int batchSize, T learningRate, string saveLocation, string saveName)
    {

        vector<int> index;
        for (int i = 0; i < groundTruths.size(); ++i)
        {
            index.push_back(i);
        }

        if(batchSize > inputLayers.size())
        {
            cout << endl;
            cout << "Batch size too large, resize to number of input data." << endl;
            batchSize = inputLayers.size();
        }

        cout << endl;

        cout << "Training start:" << endl;
        cout << "Training with:" << endl;
        cout << "Total epoch:" <<  totalEpoch << endl;
        cout << "Batch size:" <<  batchSize << endl;
        cout << "Learning rate:" <<  learningRate << endl;
        cout << endl;

        double startTime = clock();
        double lossDisplayTime = clock();
        int lossDisplayCounter = 0;
        for ( int epoch = 0; epoch < totalEpoch; epoch++)
        {
            default_random_engine randEngine(time(NULL));
            shuffle(begin(index), end(index), randEngine);



            for (int i = 0; i < inputLayers.size() / batchSize + 1; ++i)
            {
                vector<Matrix<T>> batchInputLayers;
                vector<Matrix<T>> batchGroundTruths;

                for (int j = 0; j < batchSize && (i * batchSize + j) < inputLayers.size(); ++j)
                {
                    batchInputLayers.push_back(inputLayers[index[i * batchSize + j]]);
                    batchGroundTruths.push_back(groundTruths[index[i * batchSize + j]]);
                }

                cout << "epoch: " << epoch + 1 << " batch: " << i + 1 << endl;

                update(batchInputLayers, batchGroundTruths, learningRate);

                if(lossDisplayCounter == 30 || (clock() - lossDisplayTime) / CLOCKS_PER_SEC > 20 || i == inputLayers.size() / batchSize)
                {
                    cout << endl;
                    T _loss = loss();
                    cout << "epoch: " << epoch + 1 << " batch: " << i + 1 << endl;
                    cout  << "loss: " << _loss << endl;
                    cout << endl;
                    lossDisplayTime = clock();
                    lossDisplayCounter = 0;
                    save(saveLocation, saveName);
                }
                lossDisplayCounter ++;
            }
        }
        save(saveLocation, saveName);
        cout << endl;
        cout << "Training ended." << endl;
        cout << "Trained with:" << endl;
        cout << "Total epoch:" <<  totalEpoch << endl;
        cout << "Batch size:" <<  batchSize << endl;
        cout << "Learning rate:" <<  learningRate << endl;

        int time = (clock() - (double)startTime) / CLOCKS_PER_SEC;

        int hour = time/3600;
        int min = (time%3600) / 60;
        int sec = (time%3600) % 60;

        cout << "Used time: " << hour << " hours " << min << " mins " << sec << " secs " << endl;
        cout << endl;
        T _loss = loss();
        cout << "epoch: " << totalEpoch << " loss: " << _loss << endl;
    }

    void save(const string& location, const string& name)
    {
        string saveData;

        toSaveForm(layersNums, saveData);
        toSaveForm(weights, saveData);
        toSaveForm(biases, saveData);

        stringstream ss(location);

        string folders, folder;
        while(getline(ss, folder, '/'))
        {
            folders += folder;
            mkdir(folders.c_str(), 00700);
            folders += "/";
        }

        ofstream data(ss.str() += "/" + name + ".ann");

        data << saveData;
        data.close();
   }

    void load(const string &location)
    {
        ifstream data(location);

        if (data.is_open())
        {
            cout << "Loading data ...\n";

            string line;

            layersNums.clear();
            weights.clear();
            biases.clear();

            while(getline(data, line) && !line.empty()) {
                stringstream ss(line);
                int layersNum;
                ss >> layersNum;
                layersNums.push_back(layersNum);
            }

            loadItems(weights, data);
            loadItems(biases, data);

            data.close();
            cout << "Loading data finished.\n";
        }
        else
            cout << "Unable to open file" << '\n';
    }

    void loadItems(vector<Matrix<T>> &matrices, ifstream &data)
    {
        string line;
        getline(data, line);
        stringstream ss(line);
        int count;
        ss >> count;
        loadMatrices(matrices,data, count);
    }

    void loadMatrices(vector<Matrix<T>> &matrices, ifstream &data, int count)
    {
        string line;
        getline(data, line);
        for (int i = 0; i < count; ++i)
        {
            vector<vector<T>> matrixCol;

            while(getline(data, line))
            {

                if(line.empty()) break;
                vector<T> matrixRow;
                T value;
                stringstream ss(line);

                while(true)
                {
                    ss >> value;
                    if(ss.fail()) break;
                    matrixRow.push_back(value);
                }
                matrixCol.push_back(matrixRow);
            }
            Matrix<T> matrix(matrixCol);
            matrices.push_back(matrix);

        }

    }

};
