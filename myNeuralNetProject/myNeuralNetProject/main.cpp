#include <iostream>
#include <fstream>
#include <vector>
#include <random> // for makeing gaussDistribution
#include <iomanip> // for makeing gaussDistribution

#include <Eigen/Core>

const std::string trainImageFileName = "../../data/train-images-idx3-ubyte";
const std::string trainLabelFileName = "../../data/train-labels-idx1-ubyte";
const std::string testImageFileName = "../../data/t10k-images-idx3-ubyte";
const std::string testLabelFileName = "../../data/t10k-labels-idx1-ubyte";
std::vector<std::string> train_dat = { trainImageFileName, trainLabelFileName };
std::vector<std::string> test_dat = { testImageFileName, testLabelFileName };
const int MNIST_TRAIN_IMG_NUM = 60000;
const int MNIST_TEST_IMG_NUM = 10000;
const int MNIST_IMG_SIZE = 28 * 28;
const int SRC_HEADER_OFFSET = 0x10;
const int LABEL_HEADER_OFFSET = 0x08;
const int MNIST_NUMBER_OF_OUTPUT_CLASS = 10;
const int HIDDEN_LAYER_UNIT = 100;
const double LEARNING_RATE_WEIGHT = 0.005;
const double LEARNING_RATE_BIAS = 0.01;

enum class mnist_data_type {train, test};

class mnistParser {
private :
	std::ifstream src;
	std::ifstream label;
	std::vector<std::vector<char>> src_buf;
	std::vector<char> label_buf;
public:
	mnistParser(std::vector<std::string>& dat, mnist_data_type Type);

	Eigen::MatrixXd getInput(int index);
	Eigen::MatrixXd getAnswer(int index);
};

mnistParser::mnistParser(std::vector<std::string>& dat, mnist_data_type Type)
	: src(dat[0], std::ios::in | std::ios::binary),
	label(dat[1], std::ios::in | std::ios::binary),
	src_buf(60000, std::vector<char>(MNIST_IMG_SIZE)),
	label_buf(60000) 
{
	if (!src || !label)
		throw "[ERROR] Cannnot Open File";

	src.seekg(SRC_HEADER_OFFSET);
	label.seekg(LABEL_HEADER_OFFSET);

	if (Type == mnist_data_type::train) {
		src_buf.resize(MNIST_TRAIN_IMG_NUM);
		for (auto i = 0; i < MNIST_TRAIN_IMG_NUM; i++)
			src.read(&src_buf[i][0], MNIST_IMG_SIZE);

		label.read(&label_buf.front(), MNIST_TRAIN_IMG_NUM);
	}
	else if (Type == mnist_data_type::test) {
		label_buf.resize(MNIST_TEST_IMG_NUM);
		for (auto i = 0; i < MNIST_TEST_IMG_NUM; i++)
			src.read(&src_buf[i][0], MNIST_IMG_SIZE);

		label.read(&label_buf.front(), MNIST_TEST_IMG_NUM);
	}
	else {
		throw "[ERROR] unknown data type";
	}
};

Eigen::MatrixXd mnistParser::getInput(int index) {
	Eigen::MatrixXd input(MNIST_IMG_SIZE +1,1);

	input(0,0) = static_cast<double>(0);
	for (auto i = 0; i < MNIST_IMG_SIZE; i++)
		input(i + 1, 0) = static_cast<unsigned char>(static_cast<unsigned char>(src_buf[index][i])) / 255.0;

	return input;
}

Eigen::MatrixXd mnistParser::getAnswer(int index) {
	Eigen::MatrixXd answer(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, 1);

	for (auto i = 1; i < MNIST_NUMBER_OF_OUTPUT_CLASS + 1; i++)
		answer(i, 0) = ((i - 1) == static_cast<unsigned char>(label_buf[index])) ? 1 : 0;

	return answer;
}

void test(
	Eigen::MatrixXd &w_2,
	Eigen::MatrixXd &w_3,
	Eigen::MatrixXd &u_2,
	Eigen::MatrixXd &u_3, 
	Eigen::MatrixXd &d,
	Eigen::MatrixXd &z,
	Eigen::MatrixXd &z_2,
	Eigen::MatrixXd &y,
	Eigen::MatrixXd &delta_2,
	Eigen::MatrixXd &delta_3,
	mnistParser &test_data
);

Eigen::MatrixXd ReLU(Eigen::MatrixXd &u) {
	Eigen::MatrixXd res = u;
	for(auto i=0;i<u.size();i++)
		res(i,0) = u(i, 0) > static_cast<double>(0) ? u(i,0) : static_cast<double>(0);

	return res;
}

Eigen::MatrixXd soft_max(Eigen::MatrixXd &u) {
	Eigen::MatrixXd res = u;
	double sum = static_cast<double>(0);
	for (auto k = 0; k < u.size(); k++)
		sum += std::exp(u(k + 1,0));
	for (auto k = 0; k < u.size(); k++)
		res(k + 1,0) = std::exp(u(k + 1,0)) / sum;
	return res;
}

double gaussDistribution(double mu, double sigma) {
	//! generate random number using Mersenne Twister 32bit ver.
	static std::mt19937 mt(static_cast<unsigned int>(time(NULL)));
	std::normal_distribution<> norm(mu, sigma);
	return norm(mt);
}

bool get_max_prob_val(Eigen::MatrixXd &y, Eigen::MatrixXd &d) {
	int max_prob_val = 0;
	double tmp_max_val = static_cast<double>(0.0);
	for (auto i = 1; i < MNIST_NUMBER_OF_OUTPUT_CLASS + 1; i++)
		if (y(i,0) > tmp_max_val) {
			max_prob_val = i;
			tmp_max_val = y(i,0);
		}

	return (d(max_prob_val,0) == 1);
}

int main(void)
{

	try {

		mnistParser train_data(train_dat, mnist_data_type::train);
		mnistParser test_data(test_dat, mnist_data_type::test);

		//! input layer
		Eigen::MatrixXd z(28 * 28 + 1, 1);

		// hidden layer
		Eigen::MatrixXd w_2(HIDDEN_LAYER_UNIT + 1, 28 * 28 + 1);
		// init w_2
		for (auto j = 0; j < HIDDEN_LAYER_UNIT + 1; j++)
			for (auto i = 0; i < 28 * 28 + 1; i++)
				w_2(j,i) = (i == 0) ? static_cast<double>(0) : static_cast<double> (gaussDistribution(0, 0.01));

		Eigen::MatrixXd u_2(HIDDEN_LAYER_UNIT + 1,1);
		Eigen::MatrixXd z_2(HIDDEN_LAYER_UNIT + 1, 1);
		Eigen::MatrixXd delta_2(HIDDEN_LAYER_UNIT + 1, 1);

		// output layer
		Eigen::MatrixXd w_3(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, HIDDEN_LAYER_UNIT + 1);
		// init w_3
		for (auto j = 0; j < MNIST_NUMBER_OF_OUTPUT_CLASS + 1; j++)
			for (auto i = 0; i < HIDDEN_LAYER_UNIT + 1; i++)
				w_3(j, i) = (i == 0) ? static_cast<double>(0) : static_cast<double> (gaussDistribution(0, 0.01));

		Eigen::MatrixXd u_3(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, 1);
		Eigen::MatrixXd delta_3(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, 1);
		Eigen::MatrixXd y(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, 1);

		// answer
		Eigen::MatrixXd d(MNIST_NUMBER_OF_OUTPUT_CLASS + 1, 1);

		int count = 0;
		int n_epoch = 100;
		int epoch_count = 0;
		while (epoch_count < n_epoch)
		{
			while (count < 60000)
			{
				// init params
				u_2.setZero();
				delta_2.setZero();
				u_3.setZero();
				delta_3.setZero();

				// set input layer and answer
				z = train_data.getInput(count);
				d = train_data.getAnswer(count);

				// forward propagation
				u_2 = w_2 * z;
				z_2 = ReLU(u_2);		
				u_3 = w_3 * z_2;
				y = soft_max(u_3);

				// back propagation
				delta_3 = y - d;

				for (int j = 0; j < delta_2.size(); j++) {
					for (int k = 1; k < delta_3.size(); k++)
						delta_2(j,0) += u_2(j,0) > 0 ? delta_3(k,0) * w_3(k,j) : 0;
				}
				// update params
				for (int j = 0; j < HIDDEN_LAYER_UNIT + 1; j++)
					for (int k = 1; k < MNIST_NUMBER_OF_OUTPUT_CLASS + 1; k++)
						w_3(k, j) -= (j == 0) ?
						LEARNING_RATE_BIAS * delta_3(k,0) :
						//LEARNING_RATE_WEIGHT * delta_3(k,0) * ReLU(u_2(j,0));
						LEARNING_RATE_WEIGHT * delta_3(k, 0) * z_2(j,0);
				for (int i = 0; i < 28 * 28 + 1; i++)
					for (int j = 1; j < HIDDEN_LAYER_UNIT + 1; j++)
						w_2(j,i) -= (i == 0) ?
						LEARNING_RATE_BIAS * delta_2(j,0) :
						LEARNING_RATE_WEIGHT * delta_2(j,0) * z(i,0);

				count++;

				if (count % 10000 == 0)
					test(w_2, w_3, u_2, u_3, d, z, z_2, y, delta_2, delta_3, test_data);
			}
			n_epoch++;
			count = 0;
		}

	}
	catch (char* str) {
		std::cout << str << std::endl;
		return -1;
	}

	return 0;
}

void test(
	Eigen::MatrixXd &w_2,
	Eigen::MatrixXd &w_3,
	Eigen::MatrixXd &u_2,
	Eigen::MatrixXd &u_3,
	Eigen::MatrixXd &d,
	Eigen::MatrixXd &z,
	Eigen::MatrixXd &z_2,
	Eigen::MatrixXd &y,
	Eigen::MatrixXd &delta_2,
	Eigen::MatrixXd &delta_3,
	mnistParser &test_data
)
{

	int count = 0;
	int n_correct_answer = 0;
	while (count < 10000)
	{
		// init param
		u_2.setZero();
		delta_2.setZero();
		u_3.setZero();
		delta_3.setZero();

		// set input layer and answer
		z = test_data.getInput(count);
		d = test_data.getAnswer(count);

		// forward propagation
		u_2 = w_2 * z;
		z_2 = ReLU(u_2);
		u_3 = w_3 * z_2;
		y = soft_max(u_3);

		// judge
		if (get_max_prob_val(y, d))
			n_correct_answer++;

		count++;
	}

	std::cout << "[INFO] Accuracy : " << static_cast<double>(n_correct_answer) / static_cast<double>(count) * 100.0 << std::endl;

}
