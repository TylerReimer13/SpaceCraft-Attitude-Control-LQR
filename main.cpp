#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <math.h>
#include <fstream>

using namespace std;
using namespace Eigen;

float dt = .01;

VectorXf solver_step(VectorXf states) {
	// states = w1, w2, w3, o1, o2, o3

	Matrix3f J = Matrix3f::Zero(3, 3);
	J(0, 0) = 230.;
	J(1, 1) = 240.;
	J(2, 2) = 25.;

	vector<float> w_data = { states[0], states[1], states[2] };
	float* w_ptr = &w_data[0];
	Map<Vector3f> w(w_ptr, 3);

	vector<float> o_data = { states[3], states[4], states[5] };
	float* o_ptr = &o_data[0];
	Map<Vector3f> o(o_ptr, 3);

	MatrixX3f So = MatrixX3f::Zero(3, 3);
	So(0, 1) = o[2];
	So(0, 2) = -o[1];
	So(1, 0) = -o[2];
	So(1, 2) = o[0];
	So(2, 0) = o[1];
	So(2, 1) = -o[0];

	MatrixX3f Go_a = (1. - (o.dot(o)) / 2) * MatrixX3f::Identity(3, 3);
	MatrixX3f Go = .5 * (Go_a - So + (o * o.transpose()));
	Vector3f o_dot = Go * w;

	MatrixX3f Sw = MatrixX3f::Zero(3, 3);
	Sw(0, 1) = w[2];
	Sw(0, 2) = -w[1];
	Sw(1, 0) = -w[2];
	Sw(1, 2) = w[0];
	Sw(2, 0) = w[1];
	Sw(2, 1) = -w[0];

	MatrixXf K = MatrixXf::Zero(3, 6);  // This comes from the 'lqr' MATLAB script
	K(0, 0) = 56.3383;
	K(0, 3) = 23.;
	K(1, 1) = 58.7878;
	K(1, 4) = 24.;
	K(2, 2) = 6.1237;
	K(2, 5) = 2.5;

	Vector3f u = -K * states;

	Vector3f w_dot = J.inverse() * Sw * J * w + u;

	o += o_dot * dt;
	w += w_dot * dt;

	vector<float> new_data = { w[0], w[1], w[2], o[0], o[1], o[2] };
	float* new_ptr = &new_data[0];
	Map<VectorXf> new_states(new_ptr, 6);
	return new_states;
}

int main() {
	vector<float> init = { 0., 0., 0., 1.1, -1.3, .5 };
	float* init_ptr = &init[0];
	Map<VectorXf> state_vec(init_ptr, 6);

	ofstream out("out.txt");
	for (int i = 0; i <= 5000; i++) {
		auto* coutbuf = cout.rdbuf();
		cout.rdbuf(out.rdbuf());

		cout << i*dt<< ", " << state_vec[0] << ", " << state_vec[1] << ", " << state_vec[2] << ", "<< state_vec[3] << ", " << state_vec[4] << ", " << state_vec[5] << endl;
		state_vec = solver_step(state_vec);
	}

}