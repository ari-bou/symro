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
param b_L{j in J, s in S} default 5;
param b_U{j in J, s in S} default 15;
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
var UE_3_N_AAB111{i in I, r in R, s in S, j in J, r1 in R};
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (x);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON12{i in I}: (sum {r in R, s in S, j in J, r1 in R} UE_3_N_AAB111[i,r,s,j,r1]) <= 0;
UE_ENV_3_N_AAB111_1{i in I, r in R, s in S, j in J, r1 in R}: (a_L[i,r] * (-1 * a_U[2,r1] * b[j,s] + -1 * b_L[j,s] * a[2,r1] + -1 * -1 * a_U[2,r1] * b_L[j,s]) + min(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s]) * a[i,r] + -1 * a_L[i,r] * min(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_2{i in I, r in R, s in S, j in J, r1 in R}: (a_U[i,r] * (-1 * a_U[2,r1] * b[j,s] + -1 * b_L[j,s] * a[2,r1] + -1 * -1 * a_U[2,r1] * b_L[j,s]) + max(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s]) * a[i,r] + -1 * a_U[i,r] * max(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_3{i in I, r in R, s in S, j in J, r1 in R}: (a_L[i,r] * (-1 * a_L[2,r1] * b[j,s] + -1 * b_U[j,s] * a[2,r1] + -1 * -1 * a_L[2,r1] * b_U[j,s]) + min(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s]) * a[i,r] + -1 * a_L[i,r] * min(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_4{i in I, r in R, s in S, j in J, r1 in R}: (a_U[i,r] * (-1 * a_L[2,r1] * b[j,s] + -1 * b_U[j,s] * a[2,r1] + -1 * -1 * a_L[2,r1] * b_U[j,s]) + max(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s]) * a[i,r] + -1 * a_U[i,r] * max(-1 * a_U[2,r1] * b_L[j,s], -1 * a_U[2,r1] * b_U[j,s], -1 * a_L[2,r1] * b_L[j,s], -1 * a_L[2,r1] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_5{i in I, r in R, s in S, j in J, r1 in R}: (a_L[2,r1] * (-1 * a_U[i,r] * b[j,s] + -1 * b_L[j,s] * a[i,r] + -1 * -1 * a_U[i,r] * b_L[j,s]) + min(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s]) * a[2,r1] + -1 * a_L[2,r1] * min(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_6{i in I, r in R, s in S, j in J, r1 in R}: (a_U[2,r1] * (-1 * a_U[i,r] * b[j,s] + -1 * b_L[j,s] * a[i,r] + -1 * -1 * a_U[i,r] * b_L[j,s]) + max(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s]) * a[2,r1] + -1 * a_U[2,r1] * max(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_7{i in I, r in R, s in S, j in J, r1 in R}: (a_L[2,r1] * (-1 * a_L[i,r] * b[j,s] + -1 * b_U[j,s] * a[i,r] + -1 * -1 * a_L[i,r] * b_U[j,s]) + min(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s]) * a[2,r1] + -1 * a_L[2,r1] * min(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_8{i in I, r in R, s in S, j in J, r1 in R}: (a_U[2,r1] * (-1 * a_L[i,r] * b[j,s] + -1 * b_U[j,s] * a[i,r] + -1 * -1 * a_L[i,r] * b_U[j,s]) + max(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s]) * a[2,r1] + -1 * a_U[2,r1] * max(-1 * a_U[i,r] * b_L[j,s], -1 * a_U[i,r] * b_U[j,s], -1 * a_L[i,r] * b_L[j,s], -1 * a_L[i,r] * b_U[j,s])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_9{i in I, r in R, s in S, j in J, r1 in R}: (b_L[j,s] * (-1 * a_U[i,r] * a[2,r1] + -1 * a_L[2,r1] * a[i,r] + -1 * -1 * a_U[i,r] * a_L[2,r1]) + min(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1]) * b[j,s] + -1 * b_L[j,s] * min(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_10{i in I, r in R, s in S, j in J, r1 in R}: (b_U[j,s] * (-1 * a_U[i,r] * a[2,r1] + -1 * a_L[2,r1] * a[i,r] + -1 * -1 * a_U[i,r] * a_L[2,r1]) + max(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1]) * b[j,s] + -1 * b_U[j,s] * max(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_11{i in I, r in R, s in S, j in J, r1 in R}: (b_L[j,s] * (-1 * a_L[i,r] * a[2,r1] + -1 * a_U[2,r1] * a[i,r] + -1 * -1 * a_L[i,r] * a_U[2,r1]) + min(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1]) * b[j,s] + -1 * b_L[j,s] * min(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
UE_ENV_3_N_AAB111_12{i in I, r in R, s in S, j in J, r1 in R}: (b_U[j,s] * (-1 * a_L[i,r] * a[2,r1] + -1 * a_U[2,r1] * a[i,r] + -1 * -1 * a_L[i,r] * a_U[2,r1]) + max(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1]) * b[j,s] + -1 * b_U[j,s] * max(-1 * a_U[i,r] * a_L[2,r1], -1 * a_U[i,r] * a_U[2,r1], -1 * a_L[i,r] * a_L[2,r1], -1 * a_L[i,r] * a_U[2,r1])) - UE_3_N_AAB111[i,r,s,j,r1] <= 0;
# endregion 

