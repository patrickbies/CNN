#pragma once

class Matrix {
public: 
	Matrix add(const Matrix& other);
	Matrix multiply(const Matrix& other);
	Matrix transpose();
	Matrix elementWiseMultiply(const Matrix& other);
};