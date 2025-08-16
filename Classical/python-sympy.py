import sys
import time
import statistics
from sympy.ntheory import factorint

# Configuration
MEASUREMENTS = 10000

def benchmark_sympy_factorization(n, num_measurements):
    """Benchmark SymPy's factorint function with high-precision timing."""
    times = []
    
    for _ in range(num_measurements):
        start = time.perf_counter()
        result = factorint(n)
        end = time.perf_counter()
        times.append(end - start)
    
    return times, result

def main():
    if len(sys.argv) != 2:
        print("Usage: ./sympy_factorize.py <integer>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: Please provide a valid integer")
        sys.exit(1)
    
    if n < 2:
        print("Error: Number must be >= 2")
        sys.exit(1)
    
    print(f"Benchmarking SymPy factorization of {n} with {MEASUREMENTS} measurements...")
    
    times, factors = benchmark_sympy_factorization(n, MEASUREMENTS)
    
    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times)
    
    print(f"Prime factors: {factors}")
    print(f"Average time: {avg_time:.8f} seconds")
    print(f"Standard deviation: {stdev_time:.8f} seconds")
    print(f"Measurements: {MEASUREMENTS}")

if __name__ == "__main__":
    main()
