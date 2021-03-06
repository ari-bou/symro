
#--------------------------------------------------------------------------------------------------
# SETS
#--------------------------------------------------------------------------------------------------
# beginregion

param S;
set SCENARIOS = 1..S;

param T;
set TIMEPERIODS = 1..T;

set RAW_MAT_STREAMS;
set INTERMEDIATE_STREAMS;
set PRODUCT_STREAMS;
set STREAMS = RAW_MAT_STREAMS union INTERMEDIATE_STREAMS union PRODUCT_STREAMS;
set FG_STREAMS within STREAMS;

set CORE_UNITS;
set SPLIT_POINTS;
set BLENDING_TANKS;
set UNITS = CORE_UNITS union SPLIT_POINTS union BLENDING_TANKS;

set INLETS within {STREAMS, UNITS};
set OUTLETS within {STREAMS, UNITS};

set PROPERTIES;

# endregion

#--------------------------------------------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------------------------------------------
# beginregion

param MIN_UNIT_TPT{UNITS} default 0;
param MAX_UNIT_TPT{UNITS} default 1000;

param SHUTDOWN{UNITS, TIMEPERIODS, SCENARIOS} binary, default 0;

param CORE_UNIT_YIELD{i in STREAMS, j in STREAMS, u in CORE_UNITS: (i,u) in INLETS and (j,u) in OUTLETS} default 1;

param DEMAND{PRODUCT_STREAMS} default 1000;
param MIN_PRODUCTION{PRODUCT_STREAMS} default 0;

param QUALITY{PROPERTIES, STREAMS} default 0;
param MIN_QUALITY{PROPERTIES, PRODUCT_STREAMS} default -Infinity;
param MAX_QUALITY{PROPERTIES, PRODUCT_STREAMS} default Infinity;

param MAX_TANK_SIZE := 200;

param MATERIAL_COST{STREAMS} default 0;
param OPERATING_COST{UNITS} default 0;

param PROBABILITY{SCENARIOS} default 1;

# endregion

#--------------------------------------------------------------------------------------------------
# VARIABLES
#--------------------------------------------------------------------------------------------------
# beginregion

var F_IN{(i,u) in INLETS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;
var F_OUT{(i,u) in OUTLETS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;

var INVENTORY{i in STREAMS, t in TIMEPERIODS, s in SCENARIOS} >= 0, <= 1000;

var INIT_INVENTORY{i in STREAMS} >= 0, <= 1000;
var UNIT_SIZE{u in CORE_UNITS} >= 0, <= 1000;
var TANK_SIZE{i in STREAMS} >= 0, <= 1000;
var TANK_LOCATION{i in STREAMS} binary;

var OPERATING_PROFIT{t in TIMEPERIODS, s in SCENARIOS} = (
        sum {j in PRODUCT_STREAMS, u in BLENDING_TANKS: (j,u) in OUTLETS} MATERIAL_COST[j] * F_OUT[j,u,t,s]
        + sum {j in FG_STREAMS, u in UNITS: (j,u) in OUTLETS} MATERIAL_COST[j] * F_OUT[j,u,t,s] 
        - sum {i in RAW_MAT_STREAMS, u in UNITS: (i,u) in INLETS} MATERIAL_COST[i] * F_IN[i,u,t,s]
        - sum {(i,u) in INLETS} OPERATING_COST[u] * F_IN[i,u,t,s]
    )
;

var CAPITAL_COST = 
    sum {i in STREAMS} (600 * TANK_LOCATION[i] + 0.005 * TANK_SIZE[i] ^ 2)
    + 1500 + 0.001 * UNIT_SIZE['CDU'] ^ 2
    + 1200 + 0.002 * UNIT_SIZE['RF'] ^ 2
    + 1300 + 0.0015 * UNIT_SIZE['CC'] ^ 2
;

# endregion

#--------------------------------------------------------------------------------------------------
# OBJECTIVES
#--------------------------------------------------------------------------------------------------
# beginregion
maximize TOTAL_PROFIT: sum {s in SCENARIOS} (PROBABILITY[s] * sum {t in TIMEPERIODS} (OPERATING_PROFIT[t,s])) - T * CAPITAL_COST;
maximize TIMEPERIOD_PROFIT{t in TIMEPERIODS, s in SCENARIOS}: PROBABILITY[s] * (OPERATING_PROFIT[t,s] - CAPITAL_COST);
# endregion

#--------------------------------------------------------------------------------------------------
# CONSTRAINTS
#--------------------------------------------------------------------------------------------------
# beginregion

# Mass Balances

UNIT_MASS_BALANCES{u in UNITS, t in TIMEPERIODS, s in SCENARIOS}: 
    sum {i in STREAMS: (i,u) in INLETS} F_IN[i,u,t,s] = sum {j in STREAMS: (j,u) in OUTLETS} F_OUT[j,u,t,s];

LINKAGE_MASS_BALANCES{i in STREAMS, u_out in UNITS, u_in in UNITS, t in TIMEPERIODS, s in SCENARIOS: (i,u_out) in OUTLETS and (i,u_in) in INLETS}:
    INVENTORY[i,t,s] = 
        if t == 1 then INIT_INVENTORY[i] + F_OUT[i,u_out,t,s] - F_IN[i,u_in,t,s]
        else INVENTORY[i,t-1,s] + F_OUT[i,u_out,t,s] - F_IN[i,u_in,t,s]
;

# Unit Operation

CORE_UNIT_PRODUCT_RATES{j in STREAMS, u in CORE_UNITS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    F_OUT[j,u,t,s] = sum {i in STREAMS: (i,u) in INLETS} CORE_UNIT_YIELD[i,j,u] * F_IN[i,u,t,s];

UNIT_TPT_BOUNDS{u in UNITS, t in TIMEPERIODS, s in SCENARIOS}: 
    (1-SHUTDOWN[u,t,s]) * MIN_UNIT_TPT[u] <= sum {i in STREAMS: (i,u) in INLETS} F_IN[i,u,t,s] <= (1-SHUTDOWN[u,t,s]) * MAX_UNIT_TPT[u];

LAST_TANK_INVENTORY{i in STREAMS, t in TIMEPERIODS, s in SCENARIOS: t = T}: INVENTORY[i,t,s] = INIT_INVENTORY[i];

INIT_INVENTORY_BOUND{i in STREAMS}: INIT_INVENTORY[i] = TANK_SIZE[i] / 2;

# Production and Quality Constraints

BLENDING_PRODUCT_RATES{j in PRODUCT_STREAMS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MIN_PRODUCTION[j] <= F_OUT[j,u,t,s] <= DEMAND[j];

MIN_QUALITY_SPEC{q in PROPERTIES, j in PRODUCT_STREAMS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MIN_QUALITY[q,j] * F_OUT[j,u,t,s] <= sum {i in STREAMS: (i,u) in INLETS} QUALITY[q,i] * F_IN[i,u,t,s];
    
MAX_QUALITY_SPEC{q in PROPERTIES, j in PRODUCT_STREAMS, u in BLENDING_TANKS, t in TIMEPERIODS, s in SCENARIOS: (j,u) in OUTLETS}:
    MAX_QUALITY[q,j] * F_OUT[j,u,t,s] >= sum {i in STREAMS: (i,u) in INLETS} QUALITY[q,i] * F_IN[i,u,t,s];

# Design

UNIT_DESIGN_TPT_BOUND{u in CORE_UNITS, t in TIMEPERIODS, s in SCENARIOS}: 
    sum {i in STREAMS: (i,u) in INLETS} F_IN[i,u,t,s] <= UNIT_SIZE[u];

INVENTORY_DESIGN_LEVEL_BOUND{i in STREAMS, t in TIMEPERIODS, s in SCENARIOS}: INVENTORY[i,t,s] <= TANK_SIZE[i];

INVENTORY_TANK_ALLOCATION{i in STREAMS}:
    TANK_SIZE[i] = TANK_LOCATION[i] * MAX_TANK_SIZE;
    
INVENTORY_TANK_COUNT: sum {i in STREAMS} TANK_LOCATION[i] <= 2;

NO_INVENTORY_TANK{i in STREAMS: i not in {'CRUDE_A', 'CRUDE_B', 'CRUDE_C', 'SRG', 'SRN', 'SRDS', 'SRFO', 'RFG', 'CCG', 'CCFO'}}:
    TANK_LOCATION[i] = 0;

# endregion
