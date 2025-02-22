# Quantum Computer Simulator

This is an open-source project for simulating quantum computers using Python. The repository contains several versions of the simulator, each offering different functionalities and levels of complexity.

---

## Overview

- **quantum_sim.py**  
  A simple quantum register simulator.
  - **Structure:** Simulates a single register.
  - **Commands:** Supports basic quantum operations such as H, X, Z, CNOT, and MEASURE.
  - **Input Format:** .qp files (very basic command set).
  - **Default Qubit Count:** 3 (adjustable via the command line).

- **quantum_sim2.py**  
  An extended simulator with additional features.
  - **Structure:** Simulates a single register, enhanced with macro definitions (DEF/CALL).
  - **Commands:** In addition to H, X, and Z, it also supports Y, S, T, CZ, SWAP, as well as macro definitions and calls.
  - **Input Format:** .qp files.
  - **Default Qubit Count:** 3 (adjustable).

- **quantum-sim3qcp.py**  
  An advanced simulator that emulates a rudimentary quantum CPU.
  - **Structure:** Simulates 4 registers, each with 8 qubits, operating in parallel (using multithreading).
  - **Commands:** Supports an extended command set in .qcp files, including interactive inputs (READ, LOAD), WAIT, PRINT, as well as built-in ALU commands (ADD, SUB, AND, OR, XOR).
  - **Special Features:** Parallel execution of commands, modular CPU architecture, integrated ALU, and interactive input.

---

## Requirements

- **Python 3.x**
- **NumPy** (Install via `pip install numpy`)

---

## Installation & Usage

1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Alan107-gif/Quantum-computer-simulator.git
   cd Quantum-computer-simulator
2. Start the Simulator:
   Basic Simulator (quantum_sim.py):
   python quantum_sim.py <program_file.qp> [num_qubits]
   Example:
   python quantum_sim.py simple.qp 3
Extended Simulator (quantum_sim2.py):
   python quantum_sim2.py <program_file.qp> [num_qubits]
Quantum CPU Simulator (quantum-sim3qcp.py):
   python quantum-sim3qcp.py <program_file.qcp>
   python quantum-sim3qcp.py input_interactive.qcp
