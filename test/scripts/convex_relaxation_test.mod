# SETS AND PARAMETERS
# --------------------------------------------------------------------------------------------------
# beginregion
param x_LB default 0;
param x_UB default 1;
param y_LB default 0;
param y_UB default 1;
# endregion 


# VARIABLES
# --------------------------------------------------------------------------------------------------
# beginregion
var x >= 0, <= 1;
var y >= 0, <= 1;
var z >= 0, <= 1;
var UE_2_POS_XY;
var UE_2_NEG_XY;
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (0);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON2: (-1) * if (-1) >= 0 then UE_2_POS_XY else -1 * UE_2_NEG_XY <= 0;
UE_BOUND_2_POS_XY_1: (x_LB * y + y_LB * x + -1 * x_LB * y_LB) - UE_2_POS_XY <= 0;
UE_BOUND_2_POS_XY_2: (x_UB * y + y_UB * x + -1 * x_UB * y_UB) - UE_2_POS_XY <= 0;
UE_BOUND_2_NEG_XY_1: (-1 * x_UB * y + -1 * y_LB * x + -1 * -1 * x_UB * y_LB) - UE_2_NEG_XY <= 0;
UE_BOUND_2_NEG_XY_2: (-1 * x_LB * y + -1 * y_UB * x + -1 * -1 * x_LB * y_UB) - UE_2_NEG_XY <= 0;
# endregion 

