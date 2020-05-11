#include <iostream>

using namespace std;

/*******************
 * Making this static, will make dynamic later
*   w[rows][cols] ||  w[2][4]
*
*----------------------------------------     
*      |  col0 | col1 | col2 | col3 |     
*      |  ----   ----   ----   ----
  row0 |
* row1 |
* _______________________________________
*
********************/

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
