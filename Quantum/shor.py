# --- Shor's Algorithm Implementation w/ selectable backend ---
# Original Source: https://github.com/borjan-val/qiskit-shor-demo
# Author: borjan-val
# License: GPL-3.0
# Note: Heavily edited
# ------------------------------------
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, qasm2
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import UnitaryGate, QFT
import numpy as np
from math import gcd, floor, log
from fractions import Fraction
import random
import sys
import sympy
import time

# ---- Job tracking variables ----
job_count = 0
qpu_execution_time = 0
maxAttempts = 3 # Prevent real quantum hardware noise from looping our code too long (retry on impossible result)
total_full_time = 0

# ---- Quantum Backend Selection ----
backend = None
service = None


isExportQasm = input("Do you want to export the circuit to a Qasm2 file? (y/n): ") == "y"
if isExportQasm:
    qasm2Path = input("Enter the file path (circuit info and extension will be appended): ")
else:
    qasm2Path = None

isExportCSV = False
csvPath = None

choice = input("Select backend (1) AER, (2) IBM QPU: ")

if choice.strip() == "1":
    from qiskit_aer import AerSimulator
    backend = AerSimulator()

elif choice.strip() == "2":
    from qiskit_ibm_runtime import QiskitRuntimeService

    api_key   = input("Enter your IBM Cloud API key: ")
    QiskitRuntimeService.save_account(channel="ibm_cloud", token=api_key, instance="crn:v1:bluemix:public:quantum-computing:us-east:a/35bd0796d8dc4b539c96d8643c0675ac:ec24478f-0b54-47f2-a12f-0e747b01f46c::", overwrite=True)
    service = QiskitRuntimeService(channel="ibm_cloud")
    backends = service.backends(simulator=False, operational=True)
    backend = next(b for b in backends if b.name == "ibm_torino")

    isExportCSV = input("Do you want to export the measurement results to a CSV File? (y/n) ") == "y"
    if isExportCSV:
        csvPath = input("Enter the filepath where to save job results (file extension and circuit info will be added): ")
    import csv
    from qiskit_ibm_runtime import SamplerV2 as Sampler
else:
    raise ValueError("invalid backend choice")

def save_counts_to_csv(counts, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Bitstring', 'Count'])
        for bitstring, count in counts.items():
            writer.writerow([bitstring, count])
    print(f"Results saved to {filename}")

# ---- Gates & Circuits ----
def mod_mult_gate(b,N):
    # Generate matrix/control register that performs modular multiplication on target register y, so output will be b*y mod N
    if gcd(b,N)>1:
        print(f"Error: gcd({b},{N}) > 1")

    else:
        n = floor(log(N-1,2)) + 1 # Amount of Qubit for Target Register, see find_floor
        U = np.full((2**n,2**n),0) # Size of our matrix, fill with 0 for now, a matrix applying operations on n qubit has the size 2^n x 2^n (because each qubit, which we have n of, has two numbers (amplitudes for 0 and 1))
        for x in range(N):
            U[b*x % N][x] = 1 # We define what a value x should turn into after this matrix (b*x mod N), this operation, as per definition for quantum computers, has to be unitary to be reversable, so we dont manipulate vector length
# The row of a Matrix is applied onto the column of a vector and results in a vectors row, see:
# 0 1  0 = 00 + 1*1 = 1
# 1 0  1 = 10 + 0*1 = 0
# Input 0 was turned into 1 by setting 0, 1 to 1, and input 1 was turned into 0 by setting 1, 0 to 1. That is because value 0 is in row 0,
# Every number here is represented as a vector with only a single one in the row of its value (0-based, 3 in 4th row accessed via index 3)
# Because rows are matched columns in matrix multiplication, for a number 3 (stored as row), only column 3 is relevant as only line 3 matched with it has any value (rest becomes 0 anyways)
# Column 3 becomes the new output as only column 3, matched with line 3, can produce anything other than 0
# Look above, 0 is in line 1, 0 becomes the first row of U

        for x in range(N,2**n): U[x][x] = 1     # Possible inputs but impossible outputs (N mod N = 0, N+1 mod N = 1...)
# Quantum Transformation Matrix has to assign input/output for every possible input in order to be unitary (needed), side length 2^n but we only ever have states up to N-1 so these are unused and just assigned to themselves
        G = UnitaryGate(U); G.name = f"M_{b}"   # Declare as unitary gate, relevant to be used with quantum computer
        return G

def order_finding_circuit(a, N, n, m):
    control = QuantumRegister(m, name="X") # Register of Qubits to hold exponents x
    target  = QuantumRegister(n, name="Y") # Register of Qubits to hold result result of a^x mod N
    output  = ClassicalRegister(m, name="meas") # Register of Bits for classical binary result
    circuit = QuantumCircuit(control, target, output) # Form circuit of components
    circuit.x(target[0])

    for k, qb in enumerate(control): # Go through control bits, k always being the bits position and qb the qubit itself
        circuit.h(qb) # All exponents possible in m qubits, superposition of 0 and 1 for each bit so we have all possible numbers in that range
# This also means we can apply all exponents at once, which is the big advantage as we have exponentially less calculations to do
# The control gates now contains all binary numbers from 0 to 2^m - 1

        b = pow(a,2**k,N)   # Aim still: Find period r of F(x) = a^x mod N, such that a^r = 1 mod N
# Exponent in binary: x = x(m-1)*2^(m-1) + ... x(1)*2^1 + x(0)*2^0, a^(c+d) = a^c * a^d
# Because A*B mod N = (A mod N  B mod N) mod N, a^x mod N, a^(x(m-1)2^(m-1))  ...  a^(x(0)*2^0)) mod N = (a^(x(m-1)*2^(m-1)) mod N)  ....  (a^(x(0)*2^0)) mod N) mod N
# This is the calculation of the exponent of x in bytes, with the position k being m-1
# a^(x(k)*2^k) is either 1 or a^2k (bit is either 0 or 1)
# x*1 = 1, so where x_k != 1, we can multiply the current product with a^2k mod n
# Do this now classically to efficiently prepare a quantum gatter

        controlled_gate = mod_mult_gate(b,N).control() # Create controllable version

        circuit.compose(controlled_gate, qubits=[qb] + list(target), inplace=True) # Controlled version applied on target based on state of qb, inplace changes the current circuit
    circuit.compose(QFT(m, inverse=True), qubits=control, inplace=True) # We have applied every x up to 2^m - 1, and these values, because of mod, repeat periodically with the period r
# E.g. a^x mod N = a^(x+r) mod N = a^(x+2r) mod N
# IQFT is able to find that r, meaning it can find out at which period r this happens

    circuit.measure(control, output) # Finally get the specific value y which will then give us the period r from which we can attainthe actual factor as we know its relation is y/2m = k/r for an unknown k
    return circuit # **FINALLY** we can collapse the whole system and its superpositions etc. into a single measurable value y

def find_order(a, N):
    global job_count, qpu_execution_time, total_full_time # Changed: _global_ -> global, aer_execution_time -> qpu_execution_time
    full_start = time.time()

    n = floor(log(N - 1, 2)) + 1 # Amount of qubits needed for "target register", store any possible result of 'a^x mod N' as results are 0 to N-1
    m = 2 * n # Amount of bits in control register, which controls actions applied to target register

    circuit = order_finding_circuit(a, N, n, m) # Build the actual quantum circuit
    transpiled = transpile(circuit, backend) # Adapt the circuit to the used backend. E.g. mapping qubits to physical ones or rewriting gates into combinations of others etc.

    if isExportQasm:
# Generate QASM filename with sequential number
        qasm_filename = f"{qasm2Path}_{a}_{N}.qasm"
        print(f"Exporting circuit to {qasm_filename}")
        with open(qasm_filename, "w") as f:
            f.write(qasm2.dumps(transpiled))
    attempt = 0

    r_val = None # Define r_val to ensure it's always returned

    while attempt < maxAttempts:
        attempt += 1
        job_count += 1
        if backend.__class__.__name__ == "AerSimulator":
            start_time = time.time()
            result = backend.run(transpiled, shots=1, memory=True).result() # Execute circuit on simulation, 1 shot is enough as simulator has perfect result
            qpu_execution_time += time.time() - start_time # Changed: aer_execution_time -> qpu_execution_time; Count time for statistics

            y = int(result.get_memory()[0], 2) # Get result from quantum operation

        else: # IBM QPU
            sampler = Sampler(mode=backend)

            job     = sampler.run([transpiled], shots=1024) # Execute circuit on imperfect QPU, 1024 Shots to get more accurate result as noise on real QPU can give false results
            
            qpu_job_processing_start_time = time.time() # Start QPU job processing time
            res0    = job.result()[0]
            qpu_execution_time += time.time() - qpu_job_processing_start_time # Accumulate QPU job processing time

            counts  = res0.data.meas.get_counts()
            if isExportCSV:
                save_counts_to_csv(counts, f"{csvPath}_{a}_{N}.csv")

# If most frequent result is 0, as a^0 mod N always 1 mod N, so always 1 but a non trivial information, we take the next best bitstring, as it is likely the searched for y and we dont have to start another run for nothing, as we should already have the result
            filtered_counts = {
                p: cnt                          # For each key p and value cnt,
                for p, cnt in counts.items()    # Loop over all key-value pairs in the counts dict
                if int(p) != 0                  # And only include it if the key converted to int is not 0
            }
# Check if filtered_counts is empty, might consist of only 0
            if not filtered_counts:
                print("All measurements resulted in 0. Retrying...")
                continue  # Retry the loop to get new measurements

            most_frequent_bitstring = max(filtered_counts, key=filtered_counts.get) # Get most-occurring result

            y = int(most_frequent_bitstring, 2)

        r_candidate = Fraction(y/2**m).limit_denominator(N).denominator # Translate quantum measurement y into actual period r, 'y' is likely close to k * (2^m / r) for some integer k according to QFT, so y / 2^m ≈ k / r and we do know y and m, r is denominator is "Nenner", tries out all denominators up to N and returns closest one
        if r_candidate != 0 and pow(a, r_candidate, N) == 1: # Really a^r = 1 mod N? Else retry
            r_val = r_candidate
            break
        elif attempt == maxAttempts: # If max attempts reached and no valid r found
            print(f"Failed to find order for a={a} after {maxAttempts} attempts. Last r_candidate={r_candidate}")
            r_val = None # Or handle as per desired logic, e.g. raise error or return None

    total_full_time += time.time() - full_start
    return r_val


# ---- CLI & Factoring Logic ----
N = int(input("Integer to factor: "))

# Prime has no Prime Factors (1 and itself?)
if sympy.isprime(N):
    print(f"{N} is a prime number.")
    sys.exit()

FACTOR_FOUND = False

# Every even number has a prime factor 2
if N % 2 == 0:
    print("Even number"); d = 2; FACTOR_FOUND = True

else:
# Check if number is another number to the power of something
    for k in range(2, round(log(N,2))+1): # 2^? = number, as two is the lowest, all other powers are smaller than this largest power
        d_candidate = int(round(N ** (1/k))) # Iterate through all, find
        if d_candidate**k == N: # Check exactly
            print("Number is a power"); d = d_candidate; FACTOR_FOUND = True; break

tried_values = set() # Keep track of already calculated values, no calculations twice
while not FACTOR_FOUND:
    a = random.randint(2, N-1) # Random number "a"

    if a in tried_values:
        continue
    tried_values.add(a) # Add to already calculated values if we didnt skip

    d_common_divisor = gcd(a, N) # (ggT) between our number and "a", 1 for numbers that are "Teilerfremd", if not "Teilerfremd" we found a factor by chance

    if d_common_divisor > 1:
        # This was a classical find, not what Shor's is for, but it is a factor.
        continue # Skip luckily found factors to force quantum part

    r = find_order(a, N) # a^? ≡ 1 mod N by using "efficient and faster (subexponential)" quantum computer calculation

    if r is None or r == 0: # 0 is not a valid order
        print(f"Order finding failed or returned invalid order for a={a}. Retrying with new 'a'.")
        continue

    print(f"The order of {a} mod {N} is {r}")

# Post Processing of order
    if r % 2 == 0: # Only even orders work for a^(r/2) mod N - 1
        x = pow(a, r//2, N) - 1 # (a^(r/2) mod N - 1)
        d = gcd(x, N) # (ggT) of above value with N

        if d > 1 and d < N: # Prevent results such as factors of 15 are 15 and 1
            FACTOR_FOUND = True
        elif d == N:
            print(f"Found trivial factor d=N for a={a}, r={r}. Retrying with new 'a'.")
            pass # Retry with a new 'a' implicitly if useless factors
        # else d == 1, also retry
    # else: r is odd, also retry

if FACTOR_FOUND: # Ensure d is defined if FACTOR_FOUND is true
    q = N//d # Second factor
    print(f"Factors found: {d} and {q}")
else: # Should ideally not happen if N is not prime and not 1.
    print(f"Could not find factors for {N} after exhausting options or due to repeated failures.")

print(f"Total jobs executed: {job_count}")
print(f"Total full time: {total_full_time:.2f} seconds")
print(f"Total non-backend time: {(total_full_time - qpu_execution_time):.2f} seconds")
print(f"Total quantum runtime: {qpu_execution_time:.2f} seconds")