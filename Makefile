runable:
	chmod u+x ./run_algo.py
	chmod u+x ./run_algo.py

SA:
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a SA -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a SA -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a SA -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a SA -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a SA -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a SA -i 1 -d fcps

TS:
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a TS -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a TS -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a TS -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a TS -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a TS -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a TS -i 1 -d fcps

SHC:
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a SHC -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a SHC -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a SHC -i 1 -d lcps
	export PYTHONHASHSEED=0
	./run_algo.py -l ES -a SHC -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l MS -a SHC -i 1 -d fcps
	export PYTHONHASHSEED=0
	./run_algo.py -l HS -a SHC -i 1 -d fcps

SPATIAL:
	export PYTHONHASHSEED=0
	./run_spatial.py -l ES -d lcps -i 1
	export PYTHONHASHSEED=0
	./run_spatial.py -l MS -d lcps -i 1
	export PYTHONHASHSEED=0
	./run_spatial.py -l HS -d lcps -i 1
	export PYTHONHASHSEED=0
	./run_spatial.py -l ES -d fcps -i 1
	export PYTHONHASHSEED=0
	./run_spatial.py -l MS -d fcps -i 1
	export PYTHONHASHSEED=0
	./run_spatial.py -l HS -d fcps -i 1
