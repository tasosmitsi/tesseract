// #include <iostream>
// #include "matrix.h"

// int main()
// {
//     std::cout << "Running tests..." << std::endl;
//     // Add your test cases here

//     float_prec EKF_PINIT_data[4 * 4] = {1, 0, 0, 0,
//                                         0, 1, 0, 0,
//                                         0, 0, 1, 0,
//                                         0, 0, 0, 1};
//     Matrix EKF_PINIT(4, 4, EKF_PINIT_data);

//     EKF_PINIT.vPrint();
//     EKF_PINIT.vPrintFull();

//     std::cout << "All tests passed!" << std::endl;
//     return 0;
// }