import numpy as np
import pennylane_snowflurry.utility.debug as debug
import pennylane as qml

def test_are_matrices_equivalent():
    # trouver une nouvelle facon de comparer des matrices. 
    
    mat = qml.X(0).matrix()
    assert debug.are_matrices_equivalent(mat, mat)
    
    mat2 = 1.5 * mat
    assert debug.are_matrices_equivalent(mat, mat2)
    
    mat = qml.Y(0).matrix()
    assert not debug.are_matrices_equivalent(mat, mat2)
    
    mat2 = mat * 1.5
    assert debug.are_matrices_equivalent(mat, mat2)
    
    mat2 = qml.CNOT([0, 1]).matrix()
    assert not debug.are_matrices_equivalent(mat, mat2)
    
    mat = (1.5 + 2.7j) * mat2
    assert debug.are_matrices_equivalent(mat, mat2)