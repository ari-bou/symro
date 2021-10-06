# SETS AND PARAMETERS
# --------------------------------------------------------------------------------------------------
# beginregion
set I = {1,2,3};
set J = {4,5,6};
set K = {7,8,9};
set R = {'i','ii','iii'};
set S = {'iv','v','vi'};
set T = {'vii','viii','ix'};
param a_L{i in I, r in R} default 5;
param a_U{i in I, r in R} default 15;
param x_L default 2;
param x_U default 10;
# endregion 


# VARIABLES
# --------------------------------------------------------------------------------------------------
# beginregion
var x >= 2, <= 10;
var y >= 2, <= 10;
var z >= 2, <= 10;
var a{i in I, r in R} >= 5, <= 15;
var b{j in J, s in S} >= 5, <= 15;
var c{k in K, t in T} >= 5, <= 15;
var UE_2_N_AX11{i in I, r in R};
var UE_2_P_AX11{i in I, r in R};
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (x);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON11_E2{i in I}: (sum {r in R} -1 * (-1) * UE_2_N_AX11[i,r]) <= 0;
CON11_E1{i in I}: (sum {r in R} UE_2_P_AX11[i,r]) <= 0;
UE_ENV_2_N_AX11_1{i in I, r in R}: (-1 * a_U[i,r] * x + -1 * x_L * a[i,r] + -1 * -1 * a_U[i,r] * x_L) - UE_2_N_AX11[i,r] <= 0;
UE_ENV_2_N_AX11_2{i in I, r in R}: (-1 * a_L[i,r] * x + -1 * x_U * a[i,r] + -1 * -1 * a_L[i,r] * x_U) - UE_2_N_AX11[i,r] <= 0;
UE_ENV_2_P_AX11_1{i in I, r in R}: (a_L[i,r] * x + x_L * a[i,r] + -1 * a_L[i,r] * x_L) - UE_2_P_AX11[i,r] <= 0;
UE_ENV_2_P_AX11_2{i in I, r in R}: (a_U[i,r] * x + x_U * a[i,r] + -1 * a_U[i,r] * x_U) - UE_2_P_AX11[i,r] <= 0;
# endregion 

