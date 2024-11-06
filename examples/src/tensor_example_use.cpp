#include <iostream>
#include "tensor.h"

int main()
{
    std::cout << "Here we play around with tensors..." << std::endl;

    TensorND<int, 3, 3> tensor; // Create a 2D tensor with 2 rows and 3 columns

    // tensor.setHomogen(30);
    // // set the matrix to 0
    // tensor.setIdentity();

    // print total size of the tensor
    std::cout << "Total size of the tensor: " << tensor.getTotalSize() << std::endl;

    // print number of dimensions of the tensor
    std::cout << "Number of dimensions of the tensor: " << tensor.getNumDims() << std::endl;

    // print the shape
    std::cout << "Shape of the tensor: " << tensor.getShape() << std::endl;

    // print the tensor
    // tensor.setHomogen(30).setDiagonal(10).setIdentity().print();

    tensor.setDiagonal(10);
    tensor(0, 1) = 10;
    tensor.print();

    static size_t order[] = { 1, 0};
    std::cout << "Transpose of the tensor: " << std::endl;
    tensor.transpose(order).print();
    std::cout << "Shape of the tensor: " << tensor.getShape() << std::endl;

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

    // // Set values in the tensor
    // for (size_t i = 0; i < 2; ++i)
    // {
    //     for (size_t j = 0; j < 3; ++j)
    //     {
    //         tensor(i, j) = i * 3 + j; // Assigning values based on position
    //     }
    // }

    // print the tensor manually in for loops
    // for (size_t i = 0; i < 4; ++i)
    // {
    //     for (size_t j = 0; j < 4; ++j)
    //     {
    //         std::cout << tensor(i, j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}