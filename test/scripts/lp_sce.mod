set S = {1,2,3};

param p{S};

var x1{S} >= 0, <= 1000, := 500;
var x2{S} >= 0, <= 1000, := 500;
var y{S} >= 0, <= 1000, := 500;

minimize OBJ: sum {s in S} (x1[s] + x2[s] - 10 * p[s] * y[s]);
minimize OBJ_SUB{s in S}: (x1[s] + x2[s] - 10 * p[s] * y[s]);


CON1{s in S}: p[s] * y[s] - x1[s] <= 0;
CON2{s in S}: p[s] * x2[s] + x1[s] - 15 = 0;
CON3{s in S}: 0.1 - x2[s] <= 0; 
NON_ANT{s in S}: y[s] - y[1] = 0;
