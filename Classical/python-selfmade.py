import sys
import time
import statistics
import math

# Configuration
MEASUREMENTS = 10000

def trial_division_factorize(n):
    """
    Prime factorization using optimized trial division algorithm.
    Only tests divisors up to sqrt(n) for efficiency.
    """
    if n < 2:
        return []
    
    factors = []
    
    # Handle factor 2 separately for efficiency
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Test odd divisors from 3 up to sqrt(n)
    divisor = 3
    sqrt_n = int(math.sqrt(n)) + 1
    
    while divisor <= sqrt_n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
            sqrt_n = int(math.sqrt(n)) + 1  # Update sqrt after division
        divisor += 2  # Only test odd numbers
    
    # If n > 1, then it's a prime factor
    if n > 1:
        factors.append(n)
    
    return factors

def benchmark_trial_division(n, num_measurements):
    """Benchmark trial division with high-precision timing."""
    times = []
    
    for _ in range(num_measurements):
        start = time.perf_counter()
        result = trial_division_factorize(n)
        end = time.perf_counter()
        times.append(end - start)
    
    return times, result

def main():
    if len(sys.argv) != 2:
        print("Usage: ./trial_division.py <integer>")
        sys.exit(1)
    
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Error: Please provide a valid integer")
        sys.exit(1)
    
    if n < 2:
        print("Error: Number must be >= 2")
        sys.exit(1)
    
    print(f"Benchmarking trial division factorization of {n} with {MEASUREMENTS} measurements...")
    
    times, factors = benchmark_trial_division(n, MEASUREMENTS)
    
    avg_time = statistics.mean(times)
    stdev_time = statistics.stdev(times)
    
    print(f"Prime factors: {factors}")
    print(f"Average time: {avg_time:.8f} seconds")
    print(f"Standard deviation: {stdev_time:.8f} seconds")
    print(f"Measurements: {MEASUREMENTS}")

if __name__ == "__main__":
    main()
