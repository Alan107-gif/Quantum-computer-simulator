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

    def apply_y(self, qubit_index):
        """Wendet das Y-Gate (Pauli-Y) auf das angegebene Qubit an."""
        Y = np.array([[0, -1j],
                      [1j, 0]], dtype=np.complex128)
        self._apply_1qubit_gate(Y, qubit_index)

    def apply_z(self, qubit_index):
        """Wendet das Z-Gate (Phasenumkehr) auf das angegebene Qubit an."""
        Z = np.array([[1, 0],
                      [0, -1]], dtype=np.complex128)
        self._apply_1qubit_gate(Z, qubit_index)

    def apply_s(self, qubit_index):
        """Wendet das S-Gate (Phasendrehung um π/2) auf das angegebene Qubit an."""
        S = np.array([[1, 0],
                      [0, 1j]], dtype=np.complex128)
        self._apply_1qubit_gate(S, qubit_index)

    def apply_t(self, qubit_index):
        """Wendet das T-Gate (Phasendrehung um π/4) auf das angegebene Qubit an."""
        T = np.array([[1, 0],
                      [0, np.exp(1j * np.pi/4)]], dtype=np.complex128)
        self._apply_1qubit_gate(T, qubit_index)

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

    def apply_cz(self, control_index, target_index):
        """Wendet das CZ-Gate (kontrolliertes Z) an.
           Bei |11> wird die Amplitude um -1 multipliziert."""
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
        """Wendet das SWAP-Gate an, das zwei Qubits tauscht."""
        new_state = np.zeros_like(self.state)
        num_states = len(self.state)
        for basis_state in range(num_states):
            amplitude = self.state[basis_state]
            if amplitude == 0:
                continue
            bit1 = (basis_state >> q1) & 1
            bit2 = (basis_state >> q2) & 1
            # Entferne die Bits an q1 und q2:
            basis_without = basis_state & ~(1 << q1) & ~(1 << q2)
            # Setze die Bits vertauscht:
            new_basis_state = basis_without | (bit1 << q2) | (bit2 << q1)
            new_state[new_basis_state] += amplitude
        self.state = new_state

    def measure(self):
        """
        Misst das gesamte Register.
        Der Zustand kollabiert zu einem Basisvektor, und es wird ein Bitstring zurückgegeben.
        """
        probabilities = np.abs(self.state)**2
        collapsed_index = np.random.choice(len(self.state), p=probabilities)
        new_state = np.zeros_like(self.state)
        new_state[collapsed_index] = 1.0
        self.state = new_state
        return format(collapsed_index, f'0{self.num_qubits}b')

    def __repr__(self):
        return f"Quantum State:\n{np.round(self.state, 3)}"
# -------------------------------
# 2) PROGRAMM-PARSER & -EXECUTOR (Quantum-Stat Assembler)
# -------------------------------

def expand_macros(commands, macros):
    """
    Erweitert alle CALL-Befehle durch die entsprechenden Makrobefehle.
    """
    expanded = []
    for cmd, args in commands:
        if cmd == "CALL":
            macro_name = args[0]
            if macro_name not in macros:
                raise ValueError(f"Makro '{macro_name}' nicht definiert")
            # Rekursive Expansion:
            expanded.extend(expand_macros(macros[macro_name], macros))
        else:
            expanded.append((cmd, args))
    return expanded

def parse_quantum_program(filename):
    """
    Liest Befehle aus der Datei filename und gibt eine Liste von Befehlen zurück.
    Unterstützte Befehle:
      Ein-Qubit-Gates: H, X, Y, Z, S, T  gefolgt von <q>
      Zwei-Qubit-Gates: CNOT <c> <t>, CZ <c> <t>, SWAP <q1> <q2>
      MEASURE
      Makrodefinitionen: DEF <name> ... END
      Makroaufruf: CALL <name>
    """
    main_commands = []
    macros = {}
    in_macro = False
    current_macro_name = None
    current_macro_commands = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, start=1):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        parts = line_stripped.split()
        cmd = parts[0].upper()
        
        if cmd == "DEF":
            if in_macro:
                raise ValueError(f"SyntaxError in Zeile {i}: Verschachtelte Makrodefinitionen nicht erlaubt")
            if len(parts) != 2:
                raise ValueError(f"SyntaxError in Zeile {i}: DEF benötigt einen Namen")
            in_macro = True
            current_macro_name = parts[1]
            current_macro_commands = []
            continue
        elif cmd == "END":
            if not in_macro:
                raise ValueError(f"SyntaxError in Zeile {i}: END ohne zugehöriges DEF")
            macros[current_macro_name] = current_macro_commands
            in_macro = False
            current_macro_name = None
            current_macro_commands = []
            continue
        
        # Erstelle einen Befehlstupel (Befehl, Argumente)
        if cmd in ("H", "X", "Y", "Z", "S", "T"):
            if len(parts) != 2:
                raise ValueError(f"SyntaxError in Zeile {i}: {line_stripped}")
            try:
                qubit_index = int(parts[1])
            except ValueError:
                raise ValueError(f"SyntaxError in Zeile {i}: Qubit-Index ungültig")
            command = (cmd, [qubit_index])
        elif cmd in ("CNOT", "CZ", "SWAP"):
            if len(parts) != 3:
                raise ValueError(f"SyntaxError in Zeile {i}: {line_stripped}")
            try:
                arg1 = int(parts[1])
                arg2 = int(parts[2])
            except ValueError:
                raise ValueError(f"SyntaxError in Zeile {i}: Qubit-Index ungültig")
            command = (cmd, [arg1, arg2])
        elif cmd == "MEASURE":
            if len(parts) != 1:
                raise ValueError(f"SyntaxError in Zeile {i}: {line_stripped}")
            command = (cmd, [])
        elif cmd == "CALL":
            if len(parts) != 2:
                raise ValueError(f"SyntaxError in Zeile {i}: CALL benötigt einen Makronamen")
            command = (cmd, [parts[1]])
        else:
            raise ValueError(f"SyntaxError in Zeile {i}: Unbekannter Befehl '{cmd}'")
        
        if in_macro:
            current_macro_commands.append(command)
        else:
            main_commands.append(command)
    
    if in_macro:
        raise ValueError("SyntaxError: Makrodefinition nicht abgeschlossen (fehlendes END)")
    
    # Erweitere Makroaufrufe in den Hauptbefehlen
    main_commands = expand_macros(main_commands, macros)
    return main_commands

def execute_quantum_program(filename, register, visualize=False):
    """
    Führt das Quantum-Programm aus der Datei filename auf dem Register aus.
    """
    commands = parse_quantum_program(filename)
    
    if visualize:
        print("ASCII Circuit Diagram:\n")
        print(draw_ascii_circuit(commands, register.num_qubits))
        print("-----\n")
    
    for cmd, args in commands:
        if cmd == 'H':
            register.apply_h(args[0])
        elif cmd == 'X':
            register.apply_x(args[0])
        elif cmd == 'Y':
            register.apply_y(args[0])
        elif cmd == 'Z':
            register.apply_z(args[0])
        elif cmd == 'S':
            register.apply_s(args[0])
        elif cmd == 'T':
            register.apply_t(args[0])
        elif cmd == 'CNOT':
            register.apply_cnot(args[0], args[1])
        elif cmd == 'CZ':
            register.apply_cz(args[0], args[1])
        elif cmd == 'SWAP':
            register.apply_swap(args[0], args[1])
        elif cmd == 'MEASURE':
            result = register.measure()
            print(f"Measurement result: {result}")
        else:
            raise ValueError(f"Unbekannter Befehl: {cmd}")
# -------------------------------
# 3) ASCII-CIRCUIT-VISUALISIERUNG
# -------------------------------
def draw_ascii_circuit(commands, num_qubits):
    """
    Erzeugt ein einfaches ASCII-Diagramm des Schaltkreises.
    """
    # Erstelle für jedes Qubit eine Liste mit Schaltkreisspalten.
    lines = [["---"] for _ in range(num_qubits)]
    
    for (cmd, args) in commands:
        if cmd in ("H", "X", "Y", "Z", "S", "T"):
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
        elif cmd == "CZ":
            c_idx, t_idx = args
            for q in range(num_qubits):
                if q == c_idx:
                    lines[q].append(" ● ")
                elif q == t_idx:
                    lines[q].append(" Z ")
                else:
                    lines[q].append("---")
        elif cmd == "SWAP":
            q1, q2 = args
            for q in range(num_qubits):
                if q == q1 or q == q2:
                    lines[q].append(" x ")
                else:
                    lines[q].append("---")
        elif cmd == "MEASURE":
            for q in range(num_qubits):
                lines[q].append(" M ")
        else:
            for q in range(num_qubits):
                lines[q].append(" ? ")
    
    diagram = []
    for q in range(num_qubits):
        diagram.append(f"Q{q} " + "-".join(lines[q]))
    return "\n".join(diagram)

# -------------------------------
# 4) HAUPTPROGRAMM
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantum.py <program_file.qp> [num_qubits]")
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
