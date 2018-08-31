/******************************************************************************

                            Online C Compiler.
                Code, Compile, Run and Debug C program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <stdio.h>

int main()
{
    int n = 4;
    
    int i = 0, j = 0, k = 0;
    
    double l[n][n];
    double u[n][n];
    double a[n][n];
    
    a[0][0] = 1;
    a[0][1] = 2;
    a[0][2] = 2;
    a[0][3] = 2;
    
    a[1][0] = 1;
    a[1][1] = 4;
    a[1][2] = 1;
    a[1][3] = 2;
    
    a[2][0] = 1;
    a[2][1] = 4;
    a[2][2] = 5;
    a[2][3] = 1;
    
    a[3][0] = 2;
    a[3][1] = 3;
    a[3][2] = 4;
    a[3][3] = 5;
    
    printf("Matrix A\n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%lf ", a[i][j]);
        }
        printf("\n");
    }
    
    
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (j < i)
                l[j][i] = 0;
            else
            {
                l[j][i] = a[j][i];
                for (k = 0; k < i; k++)
                {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }
        for (j = 0; j < n; j++)
        {
            if (j < i)
                u[i][j] = 0;
            else if (j == i)
                u[i][j] = 1;
            else
            {
                u[i][j] = a[i][j] / l[i][i];
                for (k = 0; k < i; k++)
                {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
    
    double detL = 1;
    double detU = 1;
    double detM = 0;
    
    printf("Matrix L\n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if(i == j) detL *= l[i][j];
            printf("%lf ", l[i][j]);
        }
        printf("\n");
    }
    
    printf("Matrix U\n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if(i == j) detU *= u[i][j];
            printf("%lf ", u[i][j]);
        }
        printf("\n");
    }
    
    printf("\nDET = %lf",detM = detL * detU);
    
    return 0;
}
