# af2rechain.py
#     Roland Dunbrack: January 2, 2026
#     With help from ChatGPT
#
# Takes input legacy-PDB format file and splits single chain AF2 model
# into separate chains given starting residue numbers for domains/regions.
# Output pdb file can then be input into ipsae.py to calculate intraprotein ipSAE
# values between domains.
#
# Example:
#    python3 af2rechain.py AGC_AKT1_model.pdb 123 141 409
#
# will create output PDB file named "AGC_AKT1_model_rechain.pdb" with
#      chain A: residues 1-122
#      chain B: residues 123-140
#      chain C: residues 141-408
#      chain D: residues 409-480 (end of original chain A)
#
# Note: by default, first chain begins on residue 1 and last chain ends on last residue of input chain A
#
# Then run ipsae.py on json file and PDB file to get intraprotein ipSAE values:
#     python3 ipsae.py AGC_AKT1_model.json AGC_AKT1_model_rechain.pdb 10 15


import sys

def parse_pdb(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line for line in lines if line.startswith('ATOM')]

def rechain_pdb(pdb_data, breakpoints):
    current_chain_id = 65  # ASCII for 'A'
    chain_breaks = [0] + breakpoints + [float('inf')]  # Adding infinity to handle last chain to the end
    output = []

    current_break_index = 0
    current_residue_number = int(pdb_data[0][22:26].strip())

    for line in pdb_data:
        residue_number = int(line[22:26].strip())
        
        if residue_number >= chain_breaks[current_break_index + 1]:
            current_break_index += 1
            current_chain_id += 1
            
        new_line = line[:21] + chr(current_chain_id) + line[22:]
        output.append(new_line)
        
        # Update the current residue number
        current_residue_number = residue_number
    
    return output

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 af2rechain.py pdbfilename.pdb 50 100 320")
        return

    pdb_filename = sys.argv[1]
    residue_breakpoints = list(map(int, sys.argv[2:]))
    pdb_data = parse_pdb(pdb_filename)
    rechained_data = rechain_pdb(pdb_data, residue_breakpoints)

    output_pdb_filename = pdb_filename.replace(".pdb","_rechain.pdb")
    OUT=open(output_pdb_filename,'w')
    
    for line in rechained_data:
        OUT.write(line)

if __name__ == "__main__":
    main()
