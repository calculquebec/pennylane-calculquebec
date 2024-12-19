"""
Contains a generic implementation of KMP search algorithm
This is not used for now, but could be used to search for equivalences in a circuit
"""

from typing import Callable
from pennylane_calculquebec.utility.optimization import T

def _compute_lps_array(pattern : list[T], compare : Callable[[T, T], bool]) ->list[int]:
    """
    Compute the longest prefix suffix (LPS) array used in KMP algorithm.
    :param pattern: The pattern for which to compute the LPS array.
    :return: The LPS array.
    """
    pattern_length = len(pattern)
    longest_prefix_suffix = [0] * pattern_length
    length = 0  # length of the previous longest prefix suffix
    index = 1
    
    while index < pattern_length:
        if compare(pattern[index], pattern[length]):
            length += 1
            longest_prefix_suffix[index] = length
            index += 1
        else:
            if length != 0:
                length = longest_prefix_suffix[length - 1]
            else:
                longest_prefix_suffix[index] = 0
                index += 1
                
    return longest_prefix_suffix

def kmp_search(array : list[T], pattern : list[T], compare : Callable[[T, T], bool]) -> int:
    """
    Perform KMP search of `pattern` in `array`.
    Args:
        array (list[T]) The array to search within.
        pattern (list[T]) The pattern to search for.
        compare (T, T -> bool) comparison operation. returns True if lhs is more than rhs
    
    Returns
        (int) the first starting index where the pattern is found in the array.
    """
    array_length = len(array)
    pattern_length = len(pattern)
    longest_prefix_suffix = _compute_lps_array(pattern, compare)
    
    array_index = 0  # index for array
    pattern_index = 0  # index for pattern
    
    while array_index < array_length:
        if compare(pattern[pattern_index], array[array_index]):
            array_index += 1
            pattern_index += 1
        
        if pattern_index == pattern_length:
            return array_index - pattern_index
        elif array_index < array_length and not compare(pattern[pattern_index], array[array_index]):
            if pattern_index != 0:
                pattern_index = longest_prefix_suffix[pattern_index - 1]
            else:
                array_index += 1
                
    return None
