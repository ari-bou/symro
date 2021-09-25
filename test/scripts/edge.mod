set NUM_SET = {1, 2, 3};
set ALPHA_SET = {'A', 'B', 'C'};
set GREEKALPHA_SET = {"alpha", "beta", "gamma"};

set NUM_ALPHA_SET = {(1, 'A'), (1, 'B'), (2, 'A'), (2, 'C'), (3, 'B'), (3, 'C')};

set INDEXED_SET{i in NUM_SET} = 1..i;
set INDEXED_SET_2{i in NUM_SET} = {(i,j) in NUM_ALPHA_SET};

var VAR_1{NUM_SET} >= 0;
var VAR_2{i in NUM_SET, (i,j) in NUM_ALPHA_SET};
var VAR_3{INDEXED_SET[1]};

minimize OBJ: 0;

CON_1{i in NUM_SET}: VAR_1[i] <= 10;

var VAR_FROM_VAR{i in NUM_SET} = VAR_1[i];

display INDEXED_SET_2;
display union {i in {1,2}} {i1 in {1,2}, j in INDEXED_SET_2[i]: i = i1};




