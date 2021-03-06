#include <iostream>
#include <fstream>
#include <thread>
#include <random>
#include <time.h>
#include <Windows.h>
using namespace std;

//нейрон
struct neuron
{
	double value; //значение нейрона
	double error; //значение ошибки

	void act() //функция активации нейрона
	{
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

//нейросеть
class network
{
public:
	int layers; //кол-во слоев
	neuron** neurons; //нейроны нейросети
	double*** weights; //[номер слоя][номер нейрона][связь со следующим слоем]
	int* size; //кол-во нейронов на каждом слое

	//производная функции активации
	double sigm_der(double x)
	{
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9))
		{
			return 0.0;
		}

		double res = x * (1.0 - x);
		return res;
	}

	//предположение об ответе
	double predict(double x)
	{
		if (x >= 0.8)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	//установка входных данных
	void set_layers(int n, int* p)
	{
		srand(time(0));

		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];

		for (int i = 0; i < n; i++)
		{
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];

			if (i < n - 1)
			{
				weights[i] = new double* [p[i]];

				for (int j = 0; j < p[i]; j++)
				{
					weights[i][j] = new double[p[i + 1]];

					for (int k = 0; k < p[i + 1]; k++)
					{
						weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];
					}
				}
			}
		}
	}

	//установка входных данных
	void set_input(double* p)
	{
		for (int i = 0; i < size[0]; i++)
		{
			neurons[0][i].value = p[i];
		}
	}

	//очистка значений нейронов в слое
	void LayersCleaner(int LayerNumber, int start, int stop)
	{
		for (int i = start; i < stop; i++)
		{
			neurons[LayerNumber][i].value = 0;
		}
	}

	//вспомогательная функция для ForwardFeed()
	void ForwardFeeder(int LayerNumber, int start, int stop)
	{
		for (int j = start; j < stop; j++)
		{
			for (int k = 0; k < size[LayerNumber - 1]; k++)
			{
				neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
			}

			neurons[LayerNumber][j].act();
		}
	}

	//изменение значений нейронов, их активация и предположение об ответе
	double ForwardFeed()
	{
		for (int i = 1; i < layers; i++)
		{
			LayersCleaner(i, 0, size[i]);
			ForwardFeeder(i, 0, size[i]);
		}

		double max = 0;
		double prediction = 0;

		for (int i = 0; i < size[layers - 1]; i++)
		{
			if (neurons[layers - 1][i].value > max)
			{
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}

		return prediction;
	}

	//подсчет значения ошибки
	void ErrorCounter(int LayerNumber, int start, int stop, double prediction, double rresult, double lr)
	{
		if (LayerNumber == layers - 1)
		{
			for (int j = start; j < stop; j++)
			{
				if (j != int(rresult))
				{
					neurons[LayerNumber][j].error = -pow(neurons[LayerNumber][j].value, 2);
				}
				else
				{
					neurons[LayerNumber][j].error = 1.0 - neurons[LayerNumber][j].value;
				}
			}
		}
		else
		{
			for (int j = start; j < stop; j++)
			{
				double error = 0.0;

				for (int k = 0; k < size[LayerNumber + 1]; k++)
				{
					error += neurons[LayerNumber + 1][k].error * weights[LayerNumber][j][k];
				}

				neurons[LayerNumber][j].error = error;
			}
		}
	}

	//обратное распространение ошибки
	void BackPropogation(double prediction, double rresult, double lr)
	{
		for (int i = layers - 1; i > 0; i--)
		{
			if (i == layers - 1)
			{
				for (int j = 0; j < size[i]; j++)
				{
					if (j != int(rresult))
					{
						neurons[i][j].error = -pow(neurons[i][j].value, 2);
					}
					else
					{
						neurons[i][j].error = 1.0 - neurons[i][j].value;
					}

				}
			}
			else
			{
				for (int j = 0; j < size[i]; j++)
				{
					double error = 0.0;

					for (int k = 0; k < size[i + 1]; k++)
					{
						error += neurons[i + 1][k].error * weights[i][j][k];
					}

					neurons[i][j].error = error;
				}
			}
		}

		for (int i = 0; i < layers - 1; i++)
		{
			for (int j = 0; j < size[i]; j++)
			{
				for (int k = 0; k < size[i + 1]; k++)
				{
					weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_der(neurons[i + 1][k].value) * neurons[i][j].value;
				}
			}
		}
	}

	//сохранение весов
	bool SaveWeights()
	{
		ofstream fout;
		fout.open("weights.txt");

		for (int i = 0; i < layers; i++)
		{
			if (i < layers - 1)
			{
				for (int j = 0; j < size[i]; j++)
				{
					for (int k = 0; k < size[i + 1]; k++)
					{
						fout << weights[i][j][k] << " ";
					}
				}
			}
		}

		fout.close();
		return 1;
	}

	//отоборажение текущего состояния нейросети
	void show()
	{
		cout << "Neural network architecture: ";

		for (int i = 0; i < layers; i++)
		{
			cout << size[i];
			if (i < layers - 1)
			{
				cout << " - ";
			}
		}
		cout << endl;

		for (int i = 0; i < layers; i++)
		{
			cout << "\n#Layer" << i + 1 << "\n\n";

			for (int j = 0; j < size[i]; j++)
			{
				cout << "Neuron #" << j + 1 << ": \n";
				cout << "Value: " << neurons[i][j].value << endl;

				if (i < layers - 1)
				{
					cout << "Weights: \n";

					for (int k = 0; k < size[i + 1]; k++)
					{
						cout << "#" << k + 1 << ": ";

						cout << weights[i][j][k] << endl;
					}
				}
			}
		}
	}
};

int main()
{
	srand(time(0));

	ifstream fin;
	ofstream fout;
	const int l = 4; //кол-во слоев
	const int input_l = 35;
	int size[l] = { input_l, 25, 15, 11 }; //заполненность слоев

	network nn;

	double input[input_l];

	int rresult;	//правильный ответ
	double result;	//нейрон с наибольшим значением
	double ra = 0;	//кол-во правлиьных ответов
	int maxra = 0;	//максимальное кол-во правильных ответов
	int maxraepoch = 0;
	const int n = 11; //кол-во элементов для теста + 1 (числа от 0 до 9)

	nn.set_layers(l, size);

	//обучение нейросети
	for (int e = 0; ra / n * 100 < 100; e++)
	{
		ra = 0;

		fin.open("lib2.txt");

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < input_l; j++)
			{
				fin >> input[j];
			}

			fin >> rresult;

			nn.set_input(input);

			result = nn.ForwardFeed();

			//nn.show();
			//cout << result << " " << rresult << " " << endl;

			if (result == rresult)
			{
				//cout << "Guessed number " << rresult << endl;
				ra++;
			}
			else
			{
				nn.BackPropogation(result, rresult, 0.6);
			}
		}

		fin.close();

		cout << "Right answers: " << ra / n * 100 << "% \t Max RA: " << double(maxra) / n * 100 << "(epoch " << maxraepoch << ")" << endl;;

		if (ra > maxra)
		{
			maxra = ra;
			maxraepoch = e;
		}

		if (maxraepoch < e - 250)
		{
			maxra = 0;
		}
	}

	nn.SaveWeights();

	fin.close();

	//применение полученных знаний
	fin.open("test2.txt");

	for (int i = 0; i < input_l; i++)
	{
		fin >> input[i];
	}

	nn.set_input(input);
	result = nn.ForwardFeed();

	//вывод ответа
	if (result != 10)
	{
		cout << "\n\n" << "I'm guessing number: " << result << "\n\n";
	}
	else
	{
		cout << "\n\n" << "I'm guessing number: " << result - 10 << "\n\n";
	}

	fin.close();

	return 0;
}
