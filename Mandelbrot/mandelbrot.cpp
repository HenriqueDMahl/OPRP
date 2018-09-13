#include <complex>
#include <iostream>
#include <omp.h>

using namespace std;

int main(){
	int max_row, max_column, max_n;
	int i,r,c,n;
	cin >> max_row;
	cin >> max_column;
	cin >> max_n;

	char **mat = (char**)malloc(sizeof(char*)*max_row);

	for (i=0; i<max_row;i++)
		mat[i]=(char*)malloc(sizeof(char)*max_column);

	#pragma omp parallel private(i,r,c,n) shared(mat) num_threads(8)
	{
		omp_set_nested(1);
		#pragma omp for schedule(dynamic)
		for(r = 0; r < max_row; ++r){
			for(c = 0; c < max_column; ++c){
				complex<float> z;
				n = 0;
				//#pragma omp for schedule(dynamic)
				while(abs(z) < 2 && ++n < max_n)
					z = pow(z, 2) + decltype(z)(
						(float)c * 2 / max_column - 1.5,
						(float)r * 2 / max_row - 1
					);
				mat[r][c]=(n == max_n ? '#' : '.');
			}
		}
	}

	for(r = 0; r < max_row; ++r){
		for(c = 0; c < max_column; ++c)
				std::cout << mat[r][c];
		cout << '\n';
	}

	return 0;
}
