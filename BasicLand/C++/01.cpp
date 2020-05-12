#include <iostream>

using namespace std;

/*********************************************************
 * Making this static, will make dynamic later
*   w[rows][cols] ||  w[2][4]
*
*----------------------------------------     
*      |  col0 | col1 | col2 | col3 |     
*      |  ----   ----   ----   ----
* row0 |
* row1 |
* _______________________________________
*
**********************************************************/

/* Will iterate through every row after base rhetoric*/ 
/* Will make a transpose function so call sizes can be brought in matching up if needed transposed */
float* dot(int input[], float weight[], int colSize) {
  float* dotProdArray = new float[colSize];

  for(int i=0;i<colSize; colSize++) {
    dotProdArray[i] = input[i] * weight[i];
  };

  return dotProdArray; /* is a  pointer allocating dyn mem will have to call delete[] once assigned */
};

int main() {
  int i[] = {1, 2, 3, 4};
  float w[2][4] = {{-0.4, 0.7, 0.2, 0.4}, {1.4, -0.8, -0.4, 0.4}};
  float b[] = {0.2, 0.4}; 
  /* @TODO bias must match the col size of the weights and this lib will need good logging for future ease */

  /******************************************************
   * int ary[][5] = {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 0} };
   *                
   * int rows =  sizeof ary / sizeof ary[0]; // 2 rows  
   *
   * int cols = sizeof ary[0] / sizeof(int); // 5 cols
   *
  ******************************************************/
   //int inputSize[2]; making a struct rather than a 2 arr maybe
   int inputSize[2];
   //inputSize[0]
   /* m x n || row by col */
   inputSize[0] = sizeof(i) / sizeof(i[0]);      /* ROW */
   inputSize[1] =  sizeof(i[0]) / sizeof(int);   /* COL */
   for(int currRow=0; currRow < 1 /*inputSize[0]*/; currRow++) {
     // dot(i[0], w[currRow][0] ,inputSize[1]);
     /* @TODO make below a print */
     cout << "  \e[0;33;40m InputShape(" << inputSize[0] << ", " << inputSize[1] << ")\e[0m" << endl;
   }

  // float output[2][1] = {
  //   {i[0]*w[0][0] + i[1]*w[0][1] + i[2]*w[0][2] + i[3]*w[0][3] + b[0]},
  //   {i[0]*w[1][0] + i[1]*w[1][1] + i[2]*w[1][2] + i[3]*w[1][3] + b[1]}
  // };

  // cout << output[0][0] << endl;
  // cout << output[1][0] << endl;

  return 0;
}
