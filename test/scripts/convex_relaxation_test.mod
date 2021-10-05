# SETS AND PARAMETERS
# --------------------------------------------------------------------------------------------------
# beginregion
param x_L default 2;
param x_U default 10;
param y_L default 2;
param y_U default 10;
# endregion 


# VARIABLES
# --------------------------------------------------------------------------------------------------
# beginregion
var x >= 2, <= 10;
var y >= 2, <= 10;
var z >= 2, <= 10;
var UE_2_N_XY11;
var UE_2_P_XY11;
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (x);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON1_E2: -1 * (-1) * UE_2_N_XY11 <= 0;
CON1_E1: UE_2_P_XY11 <= 0;
UE_ENV_2_N_XY11_1: (-1 * x_U * y + -1 * y_L * x + -1 * -1 * x_U * y_L) - UE_2_N_XY11 <= 0;
UE_ENV_2_N_XY11_2: (-1 * x_L * y + -1 * y_U * x + -1 * -1 * x_L * y_U) - UE_2_N_XY11 <= 0;
UE_ENV_2_P_XY11_1: (x_L * y + y_L * x + -1 * x_L * y_L) - UE_2_P_XY11 <= 0;
UE_ENV_2_P_XY11_2: (x_U * y + y_U * x + -1 * x_U * y_U) - UE_2_P_XY11 <= 0;
# endregion 

