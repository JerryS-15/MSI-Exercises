"""
This file contains tests that you can run to check your code.
In a terminal, navigate to this folder and run 

    pytest

to run the tests. For a more detailed output run

    pytest -v

or, to stop at the first failed test:

    pytest -x

More information can be found here: https://docs.pytest.org/en/7.1.x/reference/reference.html#command-line-flags

You are not supposed to understand or edit this file.

EDITING THIS FILE WILL NOT FIX THE PROBLEMS IN YOUR CODE!

"""


import numpy as np
import pytest

# use matplotlib backend that does not show any figures
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message=".*Matplotlib.*")  # ignore warnings from matplotlib (due to backend)
warnings.filterwarnings("ignore", message=".*invalid value*")  # ignore warnings from numpy (due to div by zero)
warnings.filterwarnings("ignore", message=".*Mean of empty slice*")  # ignore warnings from numpy (due to empty array)

# import (and run) student script
import task4 as studentScript

# import reference results
refResults: dict = np.load("exercise1_refSol.npz")

## helper functions ##
# warning logger
def logWarning(message):
    warnings.warn(message)


# check whether two arrays are the same
def checkSimilar(array: np.ndarray, reference_array: np.ndarray,varName: str) -> bool:

    # check that it is not None
    is_not_none = (array is not None)
    assert is_not_none, f"Variable {varName} is None"

    # squeeze arrays to remove singleton dimensions
    array = array.squeeze()
    reference_array = reference_array.squeeze()

    # check shapes
    # sometimes the checked arrays do have shapes like (1000,) but should have (1000,1), but thats okay
    # iterate dimensions of the two variables
    # for i in range(min([array.ndim,reference_array.ndim])):
    same_shape_as_reference = (array.shape == reference_array.shape) # to avoid long output
    assert same_shape_as_reference, f"Variable {varName} should have shape {reference_array.shape} elements in dimension but has shape {array.shape} (ignoring singleton dimensions)"

    # check values
    same_values_as_reference = np.all(np.isclose(array,reference_array,equal_nan=True)) # to avoid long output
    assert same_values_as_reference, f"Variable {varName} is not equal to its reference value" 


#############################
### GENERAL TESTS ###########
#############################

@pytest.fixture(scope="session", autouse=True)
def test_general():
    # a general test to guide the students, does not give points, and should only print warningss

    # check if all names exist
    for key in refResults.keys():
        # logWarning(key)
        if not hasattr(studentScript, key):  logWarning(f"The variable {key} does not exist.")

    # check all figures for legends and labels
    # for k in plt.get_fignums():
    #     fig = plt.figure(k)
    #     if fig.gca().get_xlabel() != "":  logWarning(f"There is no x-label in figure {k}.")
    #     if fig.gca().get_ylabel() != "": logWarning(f"There is no y-label in figure {k}.") 
    #     if fig.gca().get_legend() is not None:  logWarning(f"There is no legend on figure {k}.")

#############################
### TESTS FOR PART A ########
#############################

# check if computed correctly (3 points)
def test_A_1():
    checkSimilar(studentScript.R_SA_single,refResults['R_SA_single'],'R_SA_single')
def test_A_2():
    checkSimilar(studentScript.R_LS_single,refResults['R_LS_single'],'R_LS_single')
def test_A_3():
    checkSimilar(studentScript.R_EV_single,refResults['R_EV_single'],'R_EV_single')

# check if plotted correctly (1 point)
def test_A_4():
    assert plt.fignum_exists(1), "Figure 1 does not not exist, make sure it is created with plt.figure(1)."
    fig = plt.figure(1)
    numLines = len(fig.gca().lines)
    assert numLines == 3, f"Figure 1 should show three lines, instead there are {numLines}"


#############################
### TESTS FOR PART B ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_B_1():
    
    # check if values correct
    checkSimilar(studentScript.R_SA,refResults['R_SA'],'R_SA')
    checkSimilar(studentScript.R_EV,refResults['R_EV'],'R_EV')
    checkSimilar(studentScript.R_LS,refResults['R_LS'],'R_LS')

    # check if three figures
    for k in [2,3,4]:

        # check if created
        assert plt.fignum_exists(k), f"Figure {k} does not not exist, make sure it is created with plt.figure({k})"

        # get figures
        fig = plt.figure(k)

        # check if figure show the right number of lines
        numLines = len(fig.gca().lines)
        targetNumLines = refResults['M']
        assert  numLines == targetNumLines, f"Figure {k} should show {targetNumLines} lines, instead there are {numLines}"

#############################
### TESTS FOR PART C ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_C_1():
    
    # check if values correct
    checkSimilar(studentScript.R_SA_mean,refResults['R_SA_mean'],'R_SA_mean')
    checkSimilar(studentScript.R_EV_mean,refResults['R_EV_mean'],'R_EV_mean')
    checkSimilar(studentScript.R_LS_mean,refResults['R_LS_mean'],'R_LS_mean')


#############################
### TESTS FOR PART D ########
#############################

# check if values correct and  plotted correctly (1 point)
def test_D_1():
    
    # check if values correct
    checkSimilar(refResults['R_SA_Nmax'],studentScript.R_SA_Nmax,'R_SA_Nmax')
    checkSimilar(refResults['R_EV_Nmax'],studentScript.R_EV_Nmax,'R_EV_Nmax')
    checkSimilar(refResults['R_LS_Nmax'],studentScript.R_LS_Nmax,'R_LS_Nmax')
