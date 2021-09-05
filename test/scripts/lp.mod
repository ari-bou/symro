param p;

var x1 >= 0, <= 1000, := 500;
var x2 >= 0, <= 1000, := 500;
var y >= 0, <= 1000, := 500;

minimize OBJ: 10 * p * y - x1 - x2;

CON1: - p * y + x1 <= 0;
CON2: p * x2 + x1 - 15 = 0;
CON3: 0.1 - x2 <= 0; 
