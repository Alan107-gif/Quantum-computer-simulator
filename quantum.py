#!/usr/bin/env python3
import sys
import numpy as np

# -------------------------------
# 1) QUANTUM REGISTER (Statevector-Simulation)
# -------------------------------
class QuantumRegister:
    def __init__(self, num_qubits=3):
        """
        Initialisiert ein Register mit num_qubits Qubits.
        Der Zustand ist ein Vektor der Länge 2^n, der anfangs |0...0> entspricht.
        """
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # Start in |0...0>

    def _apply_1qubit_gate(self, gate_2x2, qubit_index):
        """Wendet ein 2x2-Gate auf das angegebene Qubit an (on-the-fly Berechnung)."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            # Bestimme, ob das Bit an qubit_index 0 oder 1 ist:
            bit = (basis_state >> qubit_index) & 1

            # Errechne den Basiszustand, wenn das Qubit auf 0 bzw. 1 gesetzt ist:
            state0 = basis_state & ~(1 << qubit_index)  # erzwingt 0
            state1 = basis_state | (1 << qubit_index)    # erzwingt 1

            # Verteilen der Amplitude entsprechend der Gate-Matrix:
            new_state[state0] += gate_2x2[bit, 0] * amplitude
            new_state[state1] += gate_2x2[bit, 1] * amplitude
        self.state = new_state

    def apply_h(self, qubit_index):
        """Wendet das Hadamard-Gate auf das angegebene Qubit an."""
        H = (1/np.sqrt(2)) * np.array([[1, 1],
                                       [1, -1]], dtype=np.complex128)
        self._apply_1qubit_gate(H, qubit_index)

    def apply_x(self, qubit_index):
        """Wendet das X-Gate (NOT) auf das angegebene Qubit an."""
        X = np.array([[0, 1],
                      [1, 0]], dtype=np.complex128)
        self._apply_1qubit_gate(X, qubit_index)

    def apply_z(self, qubit_index):
        """Wendet das Z-Gate (Phasenumkehr) auf das angegebene Qubit an."""
        Z = np.array([[1, 0],
                      [0, -1]], dtype=np.complex128)
        self._apply_1qubit_gate(Z, qubit_index)

    def apply_cnot(self, control_index, target_index):
        """Wendet das CNOT-Gate auf das Register an (on-the-fly Berechnung)."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            control_bit = (basis_state >> control_index) & 1
            if control_bit == 1:
                # Flippe das Target-Bit:
                new_basis_state = basis_state ^ (1 << target_index)
            else:
                new_basis_state = basis_state
            new_state[new_basis_state] += amplitude
        self.state = new_state

    def measure(self):
        """
        Misst das gesamte Register. 
        Der Zustand kollabiert zu einem Basisvektor, und es wird ein Bitstring zurückgegeben.
        """
        probabilities = np.abs(self.state)**2
        collapsed_index = np.random.choice(len(self.state), p=probabilities)
        # Kollabieren: der Zustand wird auf den gemessenen Basisvektor gesetzt.
        new_state = np.zeros_like(self.state)
        new_state[collapsed_index] = 1.0
        self.state = new_state
        # Rückgabe als Bitstring
        return format(collapsed_index, f'0{self.num_qubits}b')

    def __repr__(self):
        return f"Quantum State:\n{np.round(self.state, 3)}"

# -------------------------------
# 2) PROGRAMM-PARSER & -EXECUTOR (Quantum-Stat Assembler)
# -------------------------------
def execute_quantum_program(filename, register, visualize=False):
    """
    Liest Befehle aus der Datei filename und führt sie auf dem Register aus.
    Unterstützte Befehle:
      H <q>       : Hadamard-Gate auf Qubit q
      X <q>       : Pauli-X-Gate (NOT) auf Qubit q
      Z <q>       : Pauli-Z-Gate auf Qubit q
      CNOT <c> <t>: CNOT-Gate mit Steuer Qubit c und Ziel Qubit t
      MEASURE     : Misst das Register und gibt einen Bitstring zurück
    Kommentare (Zeilen, die mit '#' beginnen) und Leerzeilen werden ignoriert.
    Bei Syntaxfehlern wird eine Fehlermeldung mit Zeilennummer ausgegeben.
    """
    commands = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        parts = line_stripped.split()
        cmd = parts[0].upper()

        if cmd in ('H', 'X', 'Z'):
            if len(parts) != 2:
                raise ValueError(f"SyntaxError in line {i}: {line.strip()}")
            try:
                qubit_index = int(parts[1])
            except ValueError:
                raise ValueError(f"SyntaxError in line {i}: Qubit index invalid")
            commands.append((cmd, [qubit_index]))
        elif cmd == 'CNOT':
            if len(parts) != 3:
                raise ValueError(f"SyntaxError in line {i}: {line.strip()}")
            try:
                c_idx = int(parts[1])
                t_idx = int(parts[2])
            except ValueError:
                raise ValueError(f"SyntaxError in line {i}: Qubit index invalid")
            commands.append((cmd, [c_idx, t_idx]))
        elif cmd == 'MEASURE':
            if len(parts) != 1:
                raise ValueError(f"SyntaxError in line {i}: {line.strip()}")
            commands.append((cmd, []))
        else:
            raise ValueError(f"SyntaxError in line {i}: Unknown instruction '{cmd}'")

    if visualize:
        print("ASCII Circuit Diagram:\n")
        print(draw_ascii_circuit(commands, register.num_qubits))
        print("-----\n")

    for cmd, args in commands:
        if cmd == 'H':
            register.apply_h(args[0])
        elif cmd == 'X':
            register.apply_x(args[0])
        elif cmd == 'Z':
            register.apply_z(args[0])
        elif cmd == 'CNOT':
            register.apply_cnot(args[0], args[1])
        elif cmd == 'MEASURE':
            result = register.measure()
            print(f"Measurement result: {result}")

# -------------------------------
# 3) ASCII-CIRCUIT-VISUALISIERUNG
# -------------------------------
def draw_ascii_circuit(commands, num_qubits):
    """
    Erzeugt ein einfaches ASCII-Diagramm des Schaltkreises.
    Für 1-Qubit-Gates (H, X, Z) wird z. B. [H] angezeigt.
    Für CNOT wird auf der Steuerlinie ein ● und auf der Ziellinie ein ⊕ gesetzt.
    MEASURE wird mit M markiert.
    """
    # Erstelle für jedes Qubit eine Liste mit Schaltkreisspalten.
    lines = [["---"] for _ in range(num_qubits)]
    
    for (cmd, args) in commands:
        if cmd in ("H", "X", "Z"):
            qubit_index = args[0]
            for q in range(num_qubits):
                if q == qubit_index:
                    lines[q].append(f"[{cmd}]")
                else:
                    lines[q].append("---")
        elif cmd == "CNOT":
            c_idx, t_idx = args
            for q in range(num_qubits):
                if q == c_idx:
                    lines[q].append(" ● ")
                elif q == t_idx:
                    lines[q].append(" ⊕ ")
                else:
                    lines[q].append("---")
        elif cmd == "MEASURE":
            for q in range(num_qubits):
                lines[q].append(" M ")
        else:
            # Sollte nicht vorkommen, da wir vorher Fehler werfen.
            for q in range(num_qubits):
                lines[q].append(" ? ")
    
    # Kombiniere die Spalten zu Zeilen
    diagram = []
    for q in range(num_qubits):
        diagram.append(f"Q{q} " + "-".join(lines[q]))
    return "\n".join(diagram)

# -------------------------------
# 4) HAUPTPROGRAMM
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantum_program.py <program_file.qp> [num_qubits]")
        sys.exit(1)

    program_file = sys.argv[1]
    # Anzahl der Qubits optional als zweiter Parameter, Standard: 3
    num_qubits = int(sys.argv[2]) if len(sys.argv) >= 3 else 3

    qreg = QuantumRegister(num_qubits)
    try:
        execute_quantum_program(program_file, qreg, visualize=True)
    except ValueError as e:
        print(e)
        sys.exit(1)
