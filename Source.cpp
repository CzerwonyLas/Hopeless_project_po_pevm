#include "iostream";
#include "cmath";
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

// Запуск нейросети с параметрами Data и функцией активации a_fun
// Weights1 - веса первого слоя из 2 нейронов (на входном слое 3 нейрона)
// Weights2 - веса второго слоя из 1 нейрона (на первом слое 2 нейрона)
// Результатом будет 1 или 0
double predict(double* Weights1[2][3], double* Weights2[1][2], double a_fun(double), double Data[3])
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

	// Раскомментить, если хочется посмотреть значения всех нейронов:
	//cout << Signals1[0] << " " << Signals1[1] << " " << Signals1[2] << endl;
	//cout << Signals2[0] << " " << Signals2[1] << endl;  


	return Signals3[0];
}

int main()
{
	// Можно задавать значения 0 или 1 и смотреть на результат работы нейросети
	const double VODKA = 1;
	const double RAIN = 1;
	const double BEST_FRIEND = 0;

	// Известные веса, на которых нейросеть работает корректно: 
	// Веса первого слоя из 2 нейронов (на входном слое 3 нейрона)
	double Weights1[2][3] = { {0.25,0.25,0},{0.5,-0.4,0.9} };
	// Веса второго слоя из 1 нейрона (на первом слое 2 нейрона)
	double Weights2[1][2] = { {-1,1} };

	double Conditions[3] = { VODKA, RAIN, BEST_FRIEND };

	double result = predict(Weights1, Weights2, activation_fun, Conditions);
	cout << result << endl;

	system("pause");
	return 0;
}