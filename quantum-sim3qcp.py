#!/usr/bin/env python3
import sys
import threading
import time
import numpy as np

# =============================================================================
# 1) QUANTUMREGISTER (Statevector-Simulation für 8 Qubits)
# =============================================================================
class QuantumRegister:
    def __init__(self, num_qubits=8):
        """
        Initialisiert ein Register mit num_qubits Qubits.
        Der Zustand ist ein Vektor der Länge 2^n, der anfangs |0...0> entspricht.
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Start in |0...0>

    def _apply_1qubit_gate(self, gate, qubit_index):
        """Wendet ein 2x2-Gate auf das angegebene Qubit an."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            # Bestimme den Zustand des Ziel-Qubits:
            bit = (basis_state >> qubit_index) & 1
            # Errechne die Basiszustände, wenn das Qubit auf 0 bzw. 1 gesetzt wird:
            state0 = basis_state & ~(1 << qubit_index)
            state1 = basis_state | (1 << qubit_index)
            new_state[state0] += gate[bit, 0] * amplitude
            new_state[state1] += gate[bit, 1] * amplitude
        self.state = new_state

    def apply_h(self, qubit_index):
        """Hadamard-Gate auf das angegebene Qubit."""
        H = (1/np.sqrt(2)) * np.array([[1, 1],
                                       [1, -1]], dtype=np.complex128)
        self._apply_1qubit_gate(H, qubit_index)

    def apply_x(self, qubit_index):
        """X-Gate (NOT) auf das angegebene Qubit."""
        X = np.array([[0, 1],
                      [1, 0]], dtype=np.complex128)
        self._apply_1qubit_gate(X, qubit_index)

    def apply_y(self, qubit_index):
        """Y-Gate (Pauli-Y) auf das angegebene Qubit."""
        Y = np.array([[0, -1j],
                      [1j, 0]], dtype=np.complex128)
        self._apply_1qubit_gate(Y, qubit_index)

    def apply_z(self, qubit_index):
        """Z-Gate (Phasenumkehr) auf das angegebene Qubit."""
        Z = np.array([[1, 0],
                      [0, -1]], dtype=np.complex128)
        self._apply_1qubit_gate(Z, qubit_index)

    def apply_s(self, qubit_index):
        """S-Gate (Phasendrehung um π/2) auf das angegebene Qubit."""
        S = np.array([[1, 0],
                      [0, 1j]], dtype=np.complex128)
        self._apply_1qubit_gate(S, qubit_index)

    def apply_t(self, qubit_index):
        """T-Gate (Phasendrehung um π/4) auf das angegebene Qubit."""
        T = np.array([[1, 0],
                      [0, np.exp(1j * np.pi/4)]], dtype=np.complex128)
        self._apply_1qubit_gate(T, qubit_index)

    def apply_cnot(self, control_index, target_index):
        """CNOT-Gate: Wenn das Steuer-Qubit 1 ist, wird das Ziel-Qubit invertiert."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            control_bit = (basis_state >> control_index) & 1
            if control_bit == 1:
                new_basis_state = basis_state ^ (1 << target_index)
            else:
                new_basis_state = basis_state
            new_state[new_basis_state] += amplitude
        self.state = new_state

    def apply_cz(self, control_index, target_index):
        """CZ-Gate: Bei |11> wird die Amplitude um -1 multipliziert."""
        new_state = np.copy(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            if self.state[basis_state] == 0:
                continue
            control_bit = (basis_state >> control_index) & 1
            target_bit = (basis_state >> target_index) & 1
            if control_bit == 1 and target_bit == 1:
                new_state[basis_state] *= -1
        self.state = new_state

    def apply_swap(self, q1, q2):
        """SWAP-Gate: Vertauscht die Zustände zweier Qubits."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            bit1 = (basis_state >> q1) & 1
            bit2 = (basis_state >> q2) & 1
            basis_without = basis_state & ~(1 << q1) & ~(1 << q2)
            new_basis_state = basis_without | (bit1 << q2) | (bit2 << q1)
            new_state[new_basis_state] += amplitude
        self.state = new_state

    def measure(self):
        """
        Misst das Register. Der Zustand kollabiert und es wird ein Bitstring zurückgegeben.
        """
        probabilities = np.abs(self.state)**2
        collapsed_index = np.random.choice(len(self.state), p=probabilities)
        new_state = np.zeros_like(self.state)
        new_state[collapsed_index] = 1.0
        self.state = new_state
        return format(collapsed_index, f'0{self.num_qubits}b')

    def __repr__(self):
        return f"Quantum State:\n{np.round(self.state, 3)}"


# =============================================================================
# 2) QUANTUM-CPU: Mehrere Register und gemeinsame Steuerung
# =============================================================================
class QuantumCPU:
    def __init__(self, num_registers=4, qubits_per_register=8):
        self.num_registers = num_registers
        self.registers = [QuantumRegister(qubits_per_register) for _ in range(num_registers)]
        self.measurements = {}  # Speichert Messresultate, z.B. {0: "00011011", ...}
        self.measurements_lock = threading.Lock()

    def set_measurement(self, reg_index, value):
        with self.measurements_lock:
            self.measurements[reg_index] = value

    def get_measurement(self, reg_index):
        with self.measurements_lock:
            return self.measurements.get(reg_index, None)

    def print_all_registers(self):
        for i, reg in enumerate(self.registers):
            print(f"Register R{i} state:")
            print(reg)
            print()


# =============================================================================
# 3) Ausführung von Quantum-Befehlen für ein einzelnes Register
# =============================================================================
def execute_quantum_command(cpu, reg_index, command_line):
    tokens = command_line.split()
    if not tokens:
        return
    op = tokens[0].upper()
    reg = cpu.registers[reg_index]
    try:
        if op == "H":
            qubit = int(tokens[1])
            reg.apply_h(qubit)
        elif op == "X":
            qubit = int(tokens[1])
            reg.apply_x(qubit)
        elif op == "Y":
            qubit = int(tokens[1])
            reg.apply_y(qubit)
        elif op == "Z":
            qubit = int(tokens[1])
            reg.apply_z(qubit)
        elif op == "S":
            qubit = int(tokens[1])
            reg.apply_s(qubit)
        elif op == "T":
            qubit = int(tokens[1])
            reg.apply_t(qubit)
        elif op == "CNOT":
            q1 = int(tokens[1])
            q2 = int(tokens[2])
            reg.apply_cnot(q1, q2)
        elif op == "CZ":
            q1 = int(tokens[1])
            q2 = int(tokens[2])
            reg.apply_cz(q1, q2)
        elif op == "SWAP":
            q1 = int(tokens[1])
            q2 = int(tokens[2])
            reg.apply_swap(q1, q2)
        elif op == "MEASURE":
            result = reg.measure()
            cpu.set_measurement(reg_index, result)
            print(f"Register R{reg_index} Measurement: {result}")
        elif op == "PRINT":
            print(f"Register R{reg_index} state:")
            print(reg)
        elif op == "WAIT":
            seconds = float(tokens[1])
            time.sleep(seconds)
        else:
            print(f"Unbekannter Befehl in R{reg_index}: {command_line}")
    except Exception as e:
        print(f"Fehler in R{reg_index} beim Ausführen von '{command_line}': {e}")

def run_register_commands(cpu, reg_index, commands):
    for cmd in commands:
        execute_quantum_command(cpu, reg_index, cmd)


# =============================================================================
# 4) ALU-Einheit: Operationen auf Basis der Messwerte
# =============================================================================
def execute_alu_command(cpu, command_line):
    tokens = command_line.split()
    if len(tokens) < 3:
        print(f"Ungültiger ALU-Befehl: {command_line}")
        return
    # Format: ALU <OP> <R#> [<R#>]
    op = tokens[1].upper()
    if op == "PRINT":
        # Beispiel: ALU PRINT R0
        reg_str = tokens[2]
        if reg_str.startswith("R"):
            reg_index = int(reg_str[1:])
            meas = cpu.get_measurement(reg_index)
            if meas is not None:
                print(f"ALU: Register R{reg_index} Measurement = {meas} (int: {int(meas,2)})")
            else:
                print(f"ALU: Keine Messung für Register R{reg_index}")
    else:
        # Beispiel: ALU ADD R0 R1
        if len(tokens) < 4:
            print(f"Ungültiger ALU-Befehl: {command_line}")
            return
        reg1_str = tokens[2]
        reg2_str = tokens[3]
        if reg1_str.startswith("R") and reg2_str.startswith("R"):
            reg1 = int(reg1_str[1:])
            reg2 = int(reg2_str[1:])
            meas1 = cpu.get_measurement(reg1)
            meas2 = cpu.get_measurement(reg2)
            if meas1 is None or meas2 is None:
                print(f"ALU: Messwerte für R{reg1} oder R{reg2} nicht vorhanden")
                return
            val1 = int(meas1, 2)
            val2 = int(meas2, 2)
            if op == "ADD":
                res = val1 + val2
            elif op == "SUB":
                res = val1 - val2
            elif op == "AND":
                res = val1 & val2
            elif op == "OR":
                res = val1 | val2
            elif op == "XOR":
                res = val1 ^ val2
            else:
                print(f"ALU: Unbekannte Operation: {op}")
                return
            print(f"ALU: {op} von R{reg1} ({val1}) und R{reg2} ({val2}) = {res}")
        else:
            print(f"ALU: Ungültige Registerangabe in: {command_line}")


# =============================================================================
# 5) Parser für die QCP-Programmdatei
# =============================================================================
def parse_qcp_file(filename):
    """
    Erwartetes Dateiformat:
    
    Jeder Quantum-Befehl muss mit einem Registerpräfix beginnen, z. B.:
      R0: H 0
      R1: CNOT 0 1
      R2: MEASURE
      R3: PRINT

    ALU-Befehle beginnen mit "ALU", z. B.:
      ALU ADD R0 R1
      ALU PRINT R2
    Kommentare (beginnend mit #) und leere Zeilen werden ignoriert.
    """
    reg_commands = {i: [] for i in range(4)}
    alu_commands = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("R"):
            # Erwartetes Format: R<index>: <Befehl>
            parts = line.split(":", 1)
            if len(parts) != 2:
                print(f"Ungültige Zeile (fehlender Doppelpunkt): {line}")
                continue
            reg_part = parts[0].strip()
            command_part = parts[1].strip()
            if reg_part.startswith("R"):
                try:
                    reg_index = int(reg_part[1:])
                    if reg_index < 0 or reg_index >= 4:
                        print(f"Ungültiger Registerindex in Zeile: {line}")
                        continue
                    reg_commands[reg_index].append(command_part)
                except Exception as e:
                    print(f"Fehler beim Parsen des Registerindex in: {line} ({e})")
            else:
                print(f"Zeile ohne gültigen Registerprefix: {line}")
        elif line.upper().startswith("ALU"):
            alu_commands.append(line)
        else:
            print(f"Unbekannter Befehlstyp in Zeile: {line}")
    return reg_commands, alu_commands


# =============================================================================
# 6) Main-Funktion mit Multithreading
# =============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python quantum_sim3qcp.py <program_file.qcp>")
        sys.exit(1)
    
    filename = sys.argv[1]
    reg_commands, alu_commands = parse_qcp_file(filename)
    
    cpu = QuantumCPU(num_registers=4, qubits_per_register=8)
    
    # Starte für jedes Register einen eigenen Thread
    threads = []
    for reg_index, commands in reg_commands.items():
        t = threading.Thread(target=run_register_commands, args=(cpu, reg_index, commands))
        threads.append(t)
        t.start()
    
    # Warten, bis alle Register-Threads fertig sind
    for t in threads:
        t.join()
    
    print("\n--- ALU Operationen ---")
    for cmd in alu_commands:
        execute_alu_command(cpu, cmd)

    print("\n--- Endzustände der Register ---")
    cpu.print_all_registers()

if __name__ == "__main__":
    main()
