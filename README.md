The Symbolic Reformulation and Optimization (SymRO) package is a multi-purpose modelling/optimization toolset. The aim of this package is to provide ready-to-use, model-agnostic implementations of advanced optimization algorithms. SymRO reads a problem formulation provided by the user and builds a symbolic representation of each construct in the problem. The only input format supported at this time is a text file written in the AMPL modelling language [1]. Optimization problems are solved via a backend engine. The AMPL engine is the only backend supported at this time.


**Input Formats**
* Model file formulated in the AMPL modelling language


**Backends**
* AMPL (separate installation required)


**Features**
* Generalized Benders Decomposition (GBD) [2]
* Convex Relaxation [3] [4]


**Planned Features**

* Nonconvex GBD [5]
* Surrogate Modelling
* Pyomo support


**Acknowledgements**

SymRO was developed under the auspices of the McMaster Advanced Control Consortium (MACC). The support of the MACC is gratefully acknowledged.


**References**
1. Fourer R, Gay DM, Kernighan BW. A Modeling Language for Mathematical Programming. Management Science. 1990;36(5):519-554.
2. Geoffrion A. Generalized Benders Decomposition. Journal of Optimization Theory and Applications. 1972;10(4):237-260.
3. Maranas, C.D., Floudas, C.A. Finding all solutions of nonlinearly constrained systems of equations. J Glob Optim 7, 143–182 (1995). https://doi.org/10.1007/BF01097059
4. C.S. Adjiman, S. Dallwig, C.A. Floudas, A. Neumaier, A global optimization method, αBB, for general twice-differentiable constrained NLPs — I. Theoretical advances, Computers & Chemical Engineering, Volume 22, Issue 9, 1998, Pages 1137-1158, ISSN 0098-1354, https://doi.org/10.1016/S0098-1354(98)00027-1.
5. Li X, Tomasgard A, Barton PI. Nonconvex Generalized Benders Decomposition for Stochastic Separable Mixed-Integer Nonlinear Programs. Journal of Optimization Theory and Applications. 2011;151(3):425-454.
