#include <iostream>
#include "tensor.h"
#include "matrix.h"
#include <chrono>

using namespace std::chrono;

auto start = high_resolution_clock::now();
void tick()
{
     start = high_resolution_clock::now();
}
void tock(std::string message)
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << message << ": "
         << duration.count() << " microseconds" << std::endl;
}

void tock()
{
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken: "
         << duration.count() << " microseconds" << std::endl;
}

int main()
{
    std::cout << "Here we play around with tensors..." << std::endl;

    // print max size of the std::size_t
    std::cout << "Max size of std::size_t: " << std::numeric_limits<std::size_t>::max() << std::endl;
    TensorND<float, 2,2,2,2> tensor;  // Create a 2D tensor with 2 rows and 3 columns
    TensorND<float, 2,2> tensor2; // Create a 2D tensor with 2 rows and 3 columns

    // print the tensor
    // tensor.setHomogen(30).setDiagonal(10).setIdentity().print();

    tensor.setRandom(10, 1);
    // tensor.print();
    std::cout << "----------" << std::endl;
    tensor2.setRandom(10, 1);
    // tensor2.setSequencial().print();
    // tensor2.print();

    // auto tt = tensor + 2.4353;
    // tt = tt + 2;
    // tt.print();

    std::cout << "----------" << std::endl;
    // auto start = high_resolution_clock::now();
    // for (size_t i = 0; i < 10; ++i)
    // {
    //     // std::cout << "Iteration: " << i << std::endl;
        // auto result = TensorND<float, 2, 2,2>::einsum(tensor, tensor2, 2, 0);
    // }
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Time taken by function: "
    //      << duration.count() << " microseconds" << std::endl;
    // std::cout << "The shape of the new result tensor is: " << result.getShape() << std::endl;
    // result.print();


    // if (tt.areDimsEqual())
    // {
    //     std::cout << "The tensor is square" << std::endl;
    // }
    // else
    // {
    //     std::cout << "The tensor is not square" << std::endl;
    // }

    // static size_t order[] = { 1, 0};
    // std::cout << "Transpose of the tensor: " << std::endl;
    // tensor.transpose(order).print();
    // std::cout << "Shape of the tensor: " << tensor.getShape() << std::endl;

    // new I tensor
    // auto identityTensor = TensorND<int, 4,4,4>::I();

    // // print the tensor
    // identityTensor.print();

    // TensorND<int, 4,1> diagonalEntries; // 4 is the max size for this example

    // Get diagonal entries
    // tensor.getDiagonalEntries(diagonalEntries);

    // Print the diagonal entries
    // for (size_t i = 0; i < 4; ++i)
    // {
    //     std::cout << diagonalEntries(i, 0) << " ";
    // }

    // Print the diagonal entries
    // diagonalEntries.print();

    std::cout << "All tests passed!" << std::endl;

    Matrix<double, 10, 10> mat1(1), mat2(2), mat4(10), mat3;
    Matrix<double, 2, 3> matrix3(2);

    mat1.setIdentity()(0, 9) = 45.654;
    mat2.setIdentity();
    mat3.setHomogen(5);
    // mat1.print();
    std::cout << "----------" << std::endl;

    tick();
    // set mat to upper triangular
    auto inv = mat1.inverse();
    tock("Inverse");

    std::cout << "Inverse useing my method:" << std::endl;
    // inv.print();

    // start = high_resolution_clock::now();
    // inv = mat1.inverse_c();
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Time taken by inverse_c: "
    //      << duration.count() << " microseconds" << std::endl;
    // std::cout << "Inverse useing inverse_c:" << std::endl;
    // inv.print();

    mat2(1,2) = 3.0;
    mat2(2,1) = 3.0;

    if (mat2.isSymmetric())
    {
        std::cout << "The matrix is symmetric" << std::endl;
    }
    else
    {
        std::cout << "The matrix is not symmetric" << std::endl;
    }

    // // initialize a matrix with a 2D array
    // double initValues[2][2] = {
    //     {1.0, 5.0},
    //     {3.0, 4.0}
    // };

    // Matrix<double, 2, 2> pre_initialized = initValues;
    // // or
    // // Matrix<double, 2, 2> pre_initialized(initValues);
    // // or
    // pre_initialized = initValues;

    return 0;
}
