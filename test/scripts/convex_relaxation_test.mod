# SETS AND PARAMETERS
# --------------------------------------------------------------------------------------------------
# beginregion
param x_LB default 2;
param x_UB default 10;
param y_LB default 2;
param y_UB default 10;
param z_LB default 2;
param z_UB default 10;
# endregion 


# VARIABLES
# --------------------------------------------------------------------------------------------------
# beginregion
var x >= 2, <= 10;
var y >= 2, <= 10;
var z >= 2, <= 10;
var UE_6_POS_XYZ114;
# endregion 


# OBJECTIVES
# --------------------------------------------------------------------------------------------------
# beginregion
minimize OBJ: (x);
# endregion 


# CONSTRAINTS
# --------------------------------------------------------------------------------------------------
# beginregion
CON7: UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_1: (x_LB * ((if y_LB >= 0 then y_LB * (1 / z) else y_LB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_UB) * y + -1 * y_LB * (1 / z_UB)) + min(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB)) * x + -1 * x_LB * min(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_2: (x_UB * ((if y_LB >= 0 then y_LB * (1 / z) else y_LB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_UB) * y + -1 * y_LB * (1 / z_UB)) + max(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB)) * x + -1 * x_UB * max(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_3: (x_LB * ((if y_UB >= 0 then y_UB * (1 / z) else y_UB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_LB) * y + -1 * y_UB * (1 / z_LB)) + min(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB)) * x + -1 * x_LB * min(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_4: (x_UB * ((if y_UB >= 0 then y_UB * (1 / z) else y_UB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_LB) * y + -1 * y_UB * (1 / z_LB)) + max(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB)) * x + -1 * x_UB * max(y_LB * (1 / z_UB), y_LB * (1 / z_LB), y_UB * (1 / z_UB), y_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_5: (y_LB * ((if x_LB >= 0 then x_LB * (1 / z) else x_LB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_UB) * x + -1 * x_LB * (1 / z_UB)) + min(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB)) * y + -1 * y_LB * min(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_6: (y_UB * ((if x_LB >= 0 then x_LB * (1 / z) else x_LB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_UB) * x + -1 * x_LB * (1 / z_UB)) + max(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB)) * y + -1 * y_UB * max(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_7: (y_LB * ((if x_UB >= 0 then x_UB * (1 / z) else x_UB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_LB) * x + -1 * x_UB * (1 / z_LB)) + min(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB)) * y + -1 * y_LB * min(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_8: (y_UB * ((if x_UB >= 0 then x_UB * (1 / z) else x_UB * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + (1 / z_LB) * x + -1 * x_UB * (1 / z_LB)) + max(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB)) * y + -1 * y_UB * max(x_LB * (1 / z_UB), x_LB * (1 / z_LB), x_UB * (1 / z_UB), x_UB * (1 / z_LB))) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_9: ((1 / z_UB) * (x_LB * y + y_LB * x + -1 * x_LB * y_LB) + (if min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) >= 0 then min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (1 / z) else min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + -1 * (1 / z_UB) * min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB)) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_10: ((1 / z_LB) * (x_LB * y + y_LB * x + -1 * x_LB * y_LB) + (if max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) >= 0 then max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (1 / z) else max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + -1 * (1 / z_LB) * max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB)) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_11: ((1 / z_UB) * (x_UB * y + y_UB * x + -1 * x_UB * y_UB) + (if min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) >= 0 then min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (1 / z) else min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + -1 * (1 / z_UB) * min(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB)) - UE_6_POS_XYZ114 <= 0;
UE_BOUND_6_POS_XYZ114_12: ((1 / z_LB) * (x_UB * y + y_UB * x + -1 * x_UB * y_UB) + (if max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) >= 0 then max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (1 / z) else max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB) * (z_LB + z_UB + -1 * z) * (1 / z_LB) * (1 / z_UB)) + -1 * (1 / z_LB) * max(x_LB * y_LB, x_LB * y_UB, x_UB * y_LB, x_UB * y_UB)) - UE_6_POS_XYZ114 <= 0;
# endregion 

