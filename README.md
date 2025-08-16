# factorisation-experiments

## Overview

This repository is part of my Maturaarbeit and contains implementations of factoring algorithms in python for both classical and quantum computers, using qiskit and other libraries to achieve this task. It also contains measurements of benchmarks made using those files.

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