The Symbolic Reformulation and Optimization (SymRO) package is a multi-purpose modelling/optimization toolset. The aim of this package is to provide ready-to-use, model-agnostic implementations of advanced optimization algorithms. SymRO reads a problem formulation provided by the user, and constructs a symbolic representation of each construct in the problem. The only input format supported at this time is a text file written in the AMPL modelling language [1]. SymRO comes with a set of tools related to problem reformulation and/or optimization. To solve an optimization problem, SymRO connects to a backend engine. The AMPL engine is the only backend supported at this time.


**Input Formats**
* Model file formulated in the AMPL modelling language


**Backends**
* AMPL (separate installation required)


**Features**
* Generalized Benders Decomposition (GBD) [2]


**Planned Features**
* Convex Relaxation
* Nonconvex GBD [3]
* Surrogate Modelling
* Pyomo support


**Acknowledgements**

SymRO was developed under the auspices of the McMaster Advanced Control Consortium (MACC). The support of the MACC is gratefully acknowledged.


**References**
1. Fourer R, Gay DM, Kernighan BW. A Modeling Language for Mathematical Programming.
Management Science. 1990;36(5):519-554.
1. Geoffrion A. Generalized Benders Decomposition. Journal of Optimization Theory and
Applications. 1972;10(4):237-260.
1. Li X, Tomasgard A, Barton PI. Nonconvex Generalized Benders Decomposition for Stochas-
tic Separable Mixed-Integer Nonlinear Programs. Journal of Optimization Theory and
Applications. 2011;151(3):425-454.
