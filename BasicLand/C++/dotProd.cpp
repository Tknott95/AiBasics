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
// MAYBE DO THIS W/ SINGLE VAL ARRAYS AND THEN RELOOP?
// Testing my tf-gpu install # 2 GOING TO USE SINGLE ARR'S I TINK
// will finish later - might have to pass single arr's and use recursion
// float** dotProduct(int _vec0[][], float _vec1[][], int _r0Size, int _c0Size, int _r1Size, int _c1Size) {
//   float returnArray[_r0Size][_c0Size];
//   for(int vec0Row=0;vec0Row < _r0Size; vec0Row++) {
//     for(int vec1Row=0;vec1Row < _r1Size; vec1Row++) {
      
//       for(int vec0Col=0; vec0Col < _c0Size; vec0Col++) {
//         for(int vec1Col=0; vec1Col < _c0Size; vec1Col++) {
//           _vec0[vec0Row][vec0Col] * _vec1[vec1Row][vec1Col];
//         }
//       }

//     }
//   }

//   return returnArray;
// } 

int main() {
  int i[] = {1, 2, 3, 4};
  float w[2][4] = {{-0.4, 0.7, 0.2, 0.4}, {1.4, -0.8, -0.4, 0.4}};
  float b[] = {0.2, 0.4};

  float output[2][1] = {
    {i[0]*w[0][0] + i[1]*w[0][1] + i[2]*w[0][2] + i[3]*w[0][3] + b[0]},
    {i[0]*w[1][0] + i[1]*w[1][1] + i[2]*w[1][2] + i[3]*w[1][3] + b[1]}
  };

  cout << output[0][0] << endl;
  cout << output[1][0] << endl;

  return 0;
}
