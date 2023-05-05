# Benchmark various parts of HTA for performance optimizations

## Setup
Install HTA and benchmarking packages from the root of the repo.

```
pip install -e .
pip install -r benchmarks/requirements.txt
```

## About Pyperf

The benchmarks in this directory can be run as stand alone scripts. We use [pyperf](https://github.com/psf/pyperf) to setup, run benchmarks and measure the results.

It is worth noting how pyperf works [reference](https://pyperf.readthedocs.io/en/latest/run_benchmark.html#pyperf-architecture) -
1. Pyperf starts by spawning a first worker process (Run 1) only to calibrate the benchmark.
1. Then pyperf spawns 20 worker processes (Run 2 .. Run 21). Each worker starts by running the benchmark once to “warmup” the process, but this result is ignored in the final result. The number of processes is configurable using `-p` flag.
1. Then each worker runs the benchmark 3 times.

Even though there are multiple processes the benchmarks run serially. For a given benchmark the -
`number of runs  = (worker count) x (3 runs) x (outer loops) x (inner loops)`
where outer loop is configurable using `-l` flag. The inner loops is hard-coded in the benchmark code, we are typically using 1.

For help on options just run any benchmark script file here with `-h`.

## Example Usage

Since our functions are time consuming it is worth using lesser workers and loops.
For reasonable results use 10 workers and 1 loop -
```
python3 benchmarks/trace_load_benchmark.py -p 10 -l 1
```

If you want to quickly test the benchmark use `--debug-single-value` as
```
python3 benchmarks/trace_load_benchmark.py --debug-single-value
``
