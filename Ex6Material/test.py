import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize


# load data
data = np.load("exercise6_task2_data.npz")
v_w = data['v_w']
# N = len(v_w)
# print(N)
print(v_w)
v_plotting = np.linspace(0, 30, 1000) 
l = len(v_plotting)
print(l)

# data = np.load("exercise6_task2_refSol.npz")
# print(data.files)
# kstar = data['kstar']
# lambdaStar = data['lambdaStar']
# exceptPower = data['expectedPower']
# thetaStar = data['thetaStar']
# windSpeedProbability = data['windSpeedProbability']
# theta_test1 = data['theta_test1']
# theta_test2 = data['theta_test2']
# objectiveFunction = data['objectiveFunction_eval1']
# objectiveFunction_eval2 = data['objectiveFunction_eval2']
# k_test = data['k_test'] 
# lambda_test = data['lambda_test']
# v_test = data['v_test1']
# v_test2 = data['v_test2']
# weibullFunction_eval1 = data['weibullFunction_eval1']
# weibullFunction_eval2 = data['weibullFunction_eval2']

# print("\nkstar: ", kstar)
# print("\nlambdaStar: ", lambdaStar)
# print("\nexceptPower: ", exceptPower)
# print("\nthetaStar: ", thetaStar)
# print("\nwindSpeedProbability: ", windSpeedProbability)
# print("\ntheta_test1: ", theta_test1)
# print("\ntheta_test2: ", theta_test2)
# print("\nobjectiveFunction: ", objectiveFunction)
# print("\nobjectiveFunction_eval2: ", objectiveFunction_eval2)
# print("\nk_test: ", k_test)
# print("\nlambda_test: ", lambda_test)
# print("\nv_test: ", v_test)
# print("\nv_test2: ", v_test2)
# print("\nweibullFunction_eval1: ", weibullFunction_eval1)
# print("\nweibullFunction_eval2: ", weibullFunction_eval2)