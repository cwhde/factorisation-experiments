# factorisation-experiments

## Overview

This repository is part of my Maturaarbeit and contains implementations of factoring algorithms in python for both classical and quantum computers, using qiskit and other libraries to achieve this task. It also contains measurements of benchmarks made using those files.

## Attribution
This Shor's algorithm implementation is a heavy modified version of code originally published by @borjan-val at https://github.com/borjan-val/qiskit-shor-demo (GPL-3.0). That repo is no longer available. The version here has been heavily refactored and commented for clarity, and updated to work with real IBM Quantum hardware and current Qiskit versions.

## Project Structure

* **Classical/:** Traditional factorization implementations and benchmarks
  * Custom Python trial division algorithm
  * SymPy-based factorization for comparison
  * Shell scripts for system `factor`/`gfactor` benchmarking
* **Quantum/:** Shor's algorithm implementation with multiple backends
  * Qiskit-based implementation supporting both AER simulator and IBM QPU
  * Real quantum hardware integration for experimental measurements
* **measurements/:** Performance data and timing results from various algorithms

## Current Features

* Custom trial division implementation with optimized performance
* SymPy integration for reference classical factorization
* Shor's algorithm with selectable quantum backends (simulator/real hardware)
* Comprehensive benchmarking with statistical analysis
* CSV data export for performance comparison
* QASM circuit export for quantum algorithm analysis

## Technical Notes

* Uses high-precision timing (`time.perf_counter()`) for accurate measurements
* Supports both local quantum simulation and IBM Cloud quantum hardware
* Includes noise handling and retry logic for real quantum device limitations

## Sources
***

This repository is part of the learning process for my MA (Maturaarbeit), but it is **NOT** an official part of the MA itself.
The sources used for this project are a mix of the academic literature cited in the MA and also various additional resources (e.g., official documentation, tutorials, and community forums) that were consulted during development.
