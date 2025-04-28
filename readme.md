Lista de quehaceres para las próximas semanas: 
* Volver a correr las gráficas de representación de todos los problemas. 
* Terminar los Experimentos con SMS-EMOA (Faltan 2 sampling 3 mutations) 
* Hacer las operaciones para archivos NO-DOMINADOS 


# README 
- Ejecutar en python 3.10.8

# PLUGINS 
- **cd_diagram.py**: Plot 
- **pymoo_utils.py**: Esqueleto para poder construir las soluciones. 
- **operators.py**: Definición de los operadores utilizados sampling (muestreo), mutación y cruza (cruza no fue probado)
- **evaluation_methods.py**: Método for testing. 
- **boraddores.py**: No utilizado, apenas para test. 
- **archieving_strategies**: Algoritmo para obtener los individuos no dominados. 

# NOTEBOOKS
- **problems_Plots.ipynb**: Creación de gráficas de los problemas resueltos con NSGA-II
- **GenerateSolutions_NSGAII.ipynb**: Generación de archivos de solución utilizando NSGA-II
- **GenerateSolutions_SMSEMOA.ipynb**: Generación de archivos de solución utilizando SMS-EMOA
- **GenerateSolutions_AGEMOEA.ipynb**: Generación de archivos de solución utilizando AGE-MOEA
- **GenerateSolutions_NSGAIII.ipynb**: Generación de archivos de solución utilizando NSGA-III
- **GenerateSolutions_MOEAD.ipynb**: Generación de archivos de solución utilizando MOEA\D
- **pruebas01.ipynb**: Primer borrador, sin datos funcionales
- **test_non_dominated.ipynb**: 
- **test_optimize.ipynb**: 