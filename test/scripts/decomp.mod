set SCENARIOS := {1,2,3};
set TIMEPERIODS := {1};
set RAW_MATERIALS;
set PRODUCTS;
set MATERIALS := RAW_MATERIALS union PRODUCTS;
set LINKAGES within {i in MATERIALS, j in MATERIALS: i != j};
set QUALITIES;

param YIELD{PRODUCTS, RAW_MATERIALS};
param MIN_PROD{PRODUCTS};
param DEMAND{PRODUCTS};
param QUALITY_RAW_MAT{QUALITIES, RAW_MATERIALS};
param QUALITY_SPEC_MIN{QUALITIES, PRODUCTS};
param QUALITY_SPEC_MAX{QUALITIES, PRODUCTS};
param COST{MATERIALS, SCENARIOS};
param PROBABILITY{SCENARIOS};

var INLET{RAW_MATERIALS, TIMEPERIODS, SCENARIOS} >= 0, <= 1000, := 30;
var NODE{MATERIALS, TIMEPERIODS, SCENARIOS} >= 0, <= 1000;
var ARC{LINKAGES, TIMEPERIODS, SCENARIOS} >= 0, <= 1000;
var PROFIT{TIMEPERIODS, SCENARIOS};

maximize OBJ: sum {t in TIMEPERIODS, s in SCENARIOS} (PROBABILITY[s] * (sum {p in PRODUCTS} (COST[p,s] * NODE[p,t,s]) - sum {r in RAW_MATERIALS} (COST[r,s] * NODE[r,t,s] + 0.01 * COST[r,s] * NODE[r,t,s] ^ 1)));
maximize OBJ_SUB{t in TIMEPERIODS, s in SCENARIOS}: PROBABILITY[s] * (sum {p in PRODUCTS} (COST[p,s] * NODE[p,t,s]) - sum {r in RAW_MATERIALS} (COST[r,s] * NODE[r,t,s] + 0.01 * COST[r,s] * NODE[r,t,s] ^ 1));

INLET_FLOW_RATE{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS}: INLET[r,t,s] = NODE[r,t,s];
RAW_MAT_FLOW_RATE{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS}: sum {p in PRODUCTS} ARC[r,p,t,s] = NODE[r,t,s];
PROD_FLOW_RATE{p in PRODUCTS, t in TIMEPERIODS, s in SCENARIOS}: NODE[p,t,s] = sum {r in RAW_MATERIALS} ARC[r,p,t,s];

MIN_PRODUCTION{p in PRODUCTS, t in TIMEPERIODS, s in SCENARIOS}: MIN_PROD[p] <= NODE[p,t,s];
PROD_DEMAND{p in PRODUCTS, t in TIMEPERIODS, s in SCENARIOS}: NODE[p,t,s] <= DEMAND[p];

QUALITY_BLENDING_RULE_MIN{q in QUALITIES, p in PRODUCTS, t in TIMEPERIODS, s in SCENARIOS}: 
    QUALITY_SPEC_MIN[q,p] * NODE[p,t,s] <= sum {r in RAW_MATERIALS} QUALITY_RAW_MAT[q,r] * ARC[r,p,t,s];
QUALITY_BLENDING_RULE_MAX{q in QUALITIES, p in PRODUCTS, t in TIMEPERIODS, s in SCENARIOS}: 
    sum {r in RAW_MATERIALS} QUALITY_RAW_MAT[q,r] * ARC[r,p,t,s] <= QUALITY_SPEC_MAX[q,p] * NODE[p,t,s];

PROFIT_CALC{t in TIMEPERIODS, s in SCENARIOS}: PROFIT[t,s] = sum {p in PRODUCTS} COST[p,s] * NODE[p,t,s] - sum {r in RAW_MATERIALS} COST[r,s] * NODE[r,t,s];

NON_ANT{r in RAW_MATERIALS, t in TIMEPERIODS, s in SCENARIOS}: INLET[r,t,s] = INLET[r,t,1];