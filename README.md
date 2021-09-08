The Symbolic Reformulation and Optimization (SymRO) package is a multi-purpose modelling/optimization toolset. The aim of this package is to provide ready-to-use, model-agnostic implementations of advanced optimization algorithms. SymRO reads a problem formulation provided by the user, and constructs a symbolic representation of each construct in the problem. The only input format supported at this time is a text file written in the AMPL modelling language. SymRO comes with a set of tools related to problem reformulation and/or optimization. To solve an optimization problem, SymRO connects to a backend engine. The AMPL engine is the only backend supported at this time.

**Input Formats**
* Model file formulated in the AMPL modelling language

**Backends**
* AMPL (separate installation required)

**Features**
* Generalized Benders Decomposition (GBD)

**Planned Features**
* Convex Relaxation
* Nonconvex GBD
* Surrogate Modelling
* Pyomo support

**Acknowledgements**
SymRO was developed under the auspices of the McMaster Advanced Control Consortium (MACC). The support of the MACC is gratefully acknowledged.
