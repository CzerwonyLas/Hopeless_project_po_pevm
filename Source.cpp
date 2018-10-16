#include "iostream";
#include "cmath";
#include "time.h";
using namespace std;

// Функция активации, преобразует сигнал после суммирования в другое значение
// (Самый простой вариант - преобразует в 0 или 1)
// Является результатом работы нейрона
double activation_fun(double Signal)
{
	if (Signal < 0.5)
		return 0;
	else
		return 1;
}

double sigmoid(double X)
{
	return 1 / (1 + exp(-X));
}

double sigmoid_dx(double X)
{
	return sigmoid(X)*(1 - sigmoid(X));
}

// Запуск нейросети с параметрами Data и функцией активации a_fun
// Weights1 - веса первого слоя из 2 нейронов (на входном слое 3 нейрона)
// Weights2 - веса второго слоя из 1 нейрона (на первом слое 2 нейрона)
double predict(double Weights1[2][3], double Weights2[1][2], double a_fun(double), int Data[3])
{
	double Signals1[3] = { 0,0,0 };
	double Signals2[2] = { 0,0 };
	double Signals3[1] = { 0 };

	for (int i = 0; i < 3; ++i)
		Signals1[i] = Data[i];

	for (int i = 0; i < 2; ++i)
	{
		double Summator = 0;
		for (int j = 0; j < 3; ++j)
			Summator += Signals1[j] * Weights1[i][j];
		Signals2[i] = a_fun(Summator);
	}


	for (int i = 0; i < 1; ++i)
	{
		double Summator = 0;
		for (int j = 0; j < 2; ++j)
			Summator += Signals2[j] * Weights2[i][j];
		Signals3[i] = a_fun(Summator);
	}

	return Signals3[0];
}

void predict(double Weights1[2][3], double Weights2[1][2], double a_fun(double), int Data[3]
	, double Layer1[3], double Layer2[2], double Layer3[1])
{
	double Signals1[3] = { 0,0,0 };
	double Signals2[2] = { 0,0 };
	double Signals3[1] = { 0 };

	for (int i = 0; i < 3; ++i)
	{
		Signals1[i] = Data[i];
		Layer1[i] = Signals1[i];
	}

	for (int i = 0; i < 2; ++i)
	{
		double Summator = 0;
		for (int j = 0; j < 3; ++j)
			Summator += Signals1[j] * Weights1[i][j];
		Signals2[i] = a_fun(Summator);
		Layer2[i] = Signals2[i];
	}


	for (int i = 0; i < 1; ++i)
	{
		double Summator = 0;
		for (int j = 0; j < 2; ++j)
			Summator += Signals2[j] * Weights2[i][j];
		Signals3[i] = a_fun(Summator);
		Layer3[i] = Signals3[i];
	}
}

void backpropagation(double Weights1[2][3], double Weights2[1][2], int LearningData[3], int Expected
					,double LearningRate, double& MSE)
{
	double Layer1[3] = { 0,0,0 };
	double Layer2[2] = { 0,0 };
	double Layer3[1] = { 0 };
	predict(Weights1, Weights2, sigmoid, LearningData, Layer1, Layer2, Layer3);

	double Delta2 = 0;

	for (int i = 0; i < 1; ++i)
	{
		double Actual = Layer3[i];
		double Error = Actual - Expected;
		MSE += Error * Error;
		Delta2 = Error * sigmoid_dx(Actual);

		for (int j = 0; j < 2; ++j)
			Weights2[i][j] = Weights2[i][j] - Layer2[j] * Delta2 * LearningRate;
	}

	for (int i = 0; i < 1; ++i)
		for (int j = 0; j < 2; ++j)
		{
		double Actual = Layer2[j];
		double Error = Weights2[i][j] * Delta2;
		double Delta1 = Error * sigmoid_dx(Actual);

		for (int k = 0; k < 3; ++k)
			Weights1[j][k] = Weights1[j][k] - Layer1[k] * Delta1 * LearningRate;
		}
}

void train(double Weights1[2][3], double Weights2[1][2], int LearningKit[8][3], int AnswersKit[8],
			int Epochs, double LearningRate)
{
	for (int i = 0; i < Epochs; ++i)
		for (int j = 0; j < 8; ++j)
		{
		double MSE = 0;
		backpropagation(Weights1, Weights2, LearningKit[j], AnswersKit[j], LearningRate, MSE);
		if (i % (Epochs/100) == 0) cout << "MSE = " << MSE << endl;
		}
}


int main()
{
	std::cout.precision(10);
	std::cout.setf(std::ios::fixed);

	const int Epochs = 5000;
	const double LearningRate = 0.05;

	double Weights1[2][3] = { {-0.79,0.44,-0.43},{0.85,-0.43,0.29} };
	double Weights2[1][2] = { {-0.5,0.52} };

	//srand(time(NULL));
	//double Weights1[2][3] = { {1 / (rand() %100 +1),1 / (rand() % 100+1),1 / (rand() % 100+1)}
	//,{1 / (rand() % 100+1),1 / (rand() % 100+1),1 / (rand() % 100+1)} };
	//double Weights2[1][2] = { {1 / (rand() % 100+1),1 / (rand() % 100+1)} };

	int LearningKit[8][3] = { {0,0,0},
							 {0,0,1},
							 {0,1,0},
							 {0,1,1},
							 {1,0,0},
							 {1,0,1},
							 {1,1,0},
							 {1,1,1} };

	int AnswersKit[8] = { 0,1,1,1,1,1,1,1 };

	train(Weights1, Weights2, LearningKit, AnswersKit, Epochs, LearningRate);
	
	cout << "=================== T E S T ===================" << endl;
	for (int i = 0; i < 8; ++i)
	{
		double test = predict(Weights1, Weights2, sigmoid, LearningKit[i]);
		cout << test << endl;
	}

	system("pause");
	return 0;
}