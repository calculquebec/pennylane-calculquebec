import numpy as np
import pennylane_calculquebec.utility.debug as debug
import pennylane as qml

def test_are_matrices_equivalent():
    # trouver une nouvelle facon de comparer des matrices. 
    
    mat = qml.X(0).matrix()
    assert debug.is_equal_matrices(mat, mat)
    
    mat2 = 1.5 * mat
    assert debug.is_equal_matrices(mat, mat2)
    
    mat = qml.Y(0).matrix()
    assert not debug.is_equal_matrices(mat, mat2)
    
    mat2 = mat * 1.5
    assert debug.is_equal_matrices(mat, mat2)
    
    mat2 = qml.CNOT([0, 1]).matrix()
    assert not debug.is_equal_matrices(mat, mat2)
    
    mat = (1.5 + 2.7j) * mat2
    assert debug.is_equal_matrices(mat, mat2)