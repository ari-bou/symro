# SETS AND PARAMETERS
# --------------------------------------------------------------------------------------------------
# beginregion
set I = {1,2,3};
set J = {4,5,6};
set K = {7,8,9};
set R = {'i','ii','iii'};
set S = {'iv','v','vi'};
set T = {'vii','viii','ix'};
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
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (x);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON9_E2: -1 * log(x) <= 0;
CON9_E1: (if x_L == x_U then x else ((log(x_L)) + (((log(x_U)) - (log(x_L))) / (x_U - x_L)) * (x - (x_L)))) <= 0;
# endregion 

