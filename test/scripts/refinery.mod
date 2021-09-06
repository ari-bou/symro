
#--------------------------------------------------------------------------------------------------
# SETS
#--------------------------------------------------------------------------------------------------
# beginregion

param S;
set SCENARIOS = 1..S;

param T;
set TIMEPERIODS = 1..T;

set MATERIALS;
set RAW_MATERIALS within MATERIALS;
set BLENDING_PRODUCTS within MATERIALS;

set CORE_UNITS;
set SPLIT_POINTS;
set INVENTORY_TANKS;
set BLENDING_TANKS;
set UNITS = CORE_UNITS union SPLIT_POINTS union INVENTORY_TANKS union BLENDING_TANKS;

set INLETS within {MATERIALS, UNITS};
set OUTLETS within {MATERIALS, UNITS};

set PROPERTIES;

# endregion

#--------------------------------------------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------------------------------------------
# beginregion

param UNIT_MIN_TPT{UNITS} default 0;
param UNIT_MAX_TPT{UNITS} default 1000;

param CORE_UNIT_YIELD{i in MATERIALS, j in MATERIALS, u in CORE_UNITS: (i,u) in INLETS and (j,u) in OUTLETS} default 1;

param INIT_LEVEL{u in INVENTORY_TANKS} default 0;

param DEMAND{BLENDING_PRODUCTS};
param MIN_PRODUCTION{BLENDING_PRODUCTS};

param QUALITY{PROPERTIES, MATERIALS} default 0;
param MIN_QUALITY{PROPERTIES, BLENDING_PRODUCTS} default -Infinity;
param MAX_QUALITY{PROPERTIES, BLENDING_PRODUCTS} default Infinity;

param MATERIAL_COST{MATERIALS};
param OPERATING_COST{UNITS};

param PROBABILITY{SCENARIOS} default 1;

# endregion

#--------------------------------------------------------------------------------------------------
# VARIABLES
#--------------------------------------------------------------------------------------------------
# beginregion

var F_IN{(i,u) in INLETS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;
var F_OUT{(i,u) in INLETS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;

var LEVEL{u in INVENTORY_TANKS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;

var PROFIT{t in TIMEPERIODS, s in SCENARIOS} = (
        sum {j in BLENDING_PRODUCTS, u in BLENDING_TANKS: (j,u) in OUTLETS} MATERIAL_COST[j] * F_OUT[j,u,t,s]
        - sum {i in RAW_MATERIALS, u in UNITS: (i,u) in INLETS} MATERIAL_COST[i] * F_IN[i,u,t,s]
        - sum {(i,u) in INLETS} OPERATING_COST * F_IN[i,u,t,s]
    )
;

# endregion

#--------------------------------------------------------------------------------------------------
# OBJECTIVES
#--------------------------------------------------------------------------------------------------
# beginregion
maximize TOTAL_PROFIT: sum {s in SCENARIOS} (PROBABILITY[s] * sum {t in TIMEPERIODS} (PROFIT[t,s]));
maximize TIMEPERIOD_PROFIT{t in TIMEPERIODS, s in SCENARIOS}: PROBABILITY[s] * PROFIT[t,s];
# endregion

#--------------------------------------------------------------------------------------------------
# CONSTRAINTS
#--------------------------------------------------------------------------------------------------
# beginregion

# Mass Balances

MASS_BALANCES{u in UNITS, t in TIMEPERIODS, s in SCENARIOS: u not in INVENTORY_TANKS}: 
    sum {i in MATERIALS: (i,u) in INLETS} F_IN[i,u,t,s] = sum {j in MATERIALS: (j,u) in OUTLETS} F_OUT[j,u,t,s];

INVENTORY_TANK_MASS_BALANCES{u in INVENTORY_TANKS, t in TIMEPERIODS, s in SCENARIOS}: 
    LEVEL[u,t,s] =
        if t == 1 then (INIT_LEVEL[u] + sum {j in MATERIALS: (j,u) in OUTLETS} F_OUT[j,u,t,s] - sum {i in MATERIALS: (i,u) in INLETS} F_IN[i,u,t,s])
        else (LEVEL[u,t-1,s] + sum {j in MATERIALS: (j,u) in OUTLETS} F_OUT[j,u,t,s] - sum {i in MATERIALS: (i,u) in INLETS} F_IN[i,u,t,s])
;

# Unit Operation

CORE_UNIT_PRODUCT_RATES{j in MATERIALS, u in CORE_UNITS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    F_OUT[j,u,t,s] = sum {i in MATERIALS: (i,u) in INLETS} CORE_UNIT_YIELD[i,j,u] * F_IN[i,u,t,s];

UNIT_THROUGHPUT_BOUNDS{u in UNITS, t in TIMEPERIODS, s in SCENARIOS}: 
    UNIT_MIN_TPT[u] <= sum {i in MATERIALS: (i,u) in INLETS} F_IN[i,u,t,s] <= UNIT_MAX_TPT[u];

# Production and Quality Constraints

BLENDING_PRODUCT_RATES{j in BLENDING_PRODUCTS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MIN_PRODUCTION[j] <= F_OUT[j,u,t,s] <= DEMAND[j];

MIN_QUALITY_SPEC{q in PROPERTIES, j in BLENDING_PRODUCTS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MIN_QUALITY[q,j] * F_OUT[j,u,t,s] <= sum {i in MATERIALS: (i,u) in INLETS} QUALITY[q,i] * F_IN[i,u,t,s];
    
MAX_QUALITY_SPEC{q in PROPERTIES, j in BLENDING_PRODUCTS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MAX_QUALITY[q,j] * F_OUT[j,u,t,s] => sum {i in MATERIALS: (i,u) in INLETS} QUALITY[q,i] * F_IN[i,u,t,s];

# endregion
