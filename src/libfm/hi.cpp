#include<stdio.h>
#include<iostream>
#include<omp.h>
using namespace std;
int main()
{

	int i;
	#pragma omp parallel
	{
		cout<<"hi\n";
	}
return 0;
}
