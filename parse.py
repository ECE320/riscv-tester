"""
    Authors: Justin Mendes and Shazil Razzaq
    Date: Monday October 17, 2022
    Last Modified: Sunday October 30, 2022
    Version: v1.2

    Tests functionality of RISC-V processor up to PD4

    USAGE:  python3 parse.py [-h/--help] [-s/--skip] [-i/--instructions] [-in/--instructions-no-num] [-r/--regfile] [-m <MEM_DEPTH> / --mem <MEM_DEPTH>] [-t/--terminate/--ecall] <PATH to .trace file> <PATH to .x file>

    EX:     python3 parse.py rv32ui-p-sltiu.trace /rv32-benchmarks/individual-instructions/rv32ui-p-sltiu.x

    UPDATES:
    @justincmendes:
    - up to PD4 instead of up to PD3
    - Fixed bugs: SUB execution result and B-Type verification logic
    - Memory and Writeback stages (end-to-end)
    - Added error messages for tracing when the program encounters an error
    - Support flag args: 
    -- -i for instructions: 
        Print all instructions in RISC-V format (Default: False)
    -- -r for register file: 
        Print the state of the Register File at each [R] (Default: False)
    -- -t to terminate the program on ECALL:
        Stop checking on ECALL to avoid parsing check beyond the scope of the program
    -- -skip for skip on errors (i.e. don't stop on errors): 
        Stop execution on errors (Default: True)
    -- -m <MEM_DEPTH>: 
        Default: 1048576

    FUTURE:
    v1.x: 
        - Add support for flags (-pd1 -pd2 -pd3 -pd4 -pd5)
            To separate the parsers verification for each of the different projects
            (upto and including the project)
"""

#// TODO: ADD REGISTER FILE + DMEMORY + WRITEBACK LOGIC!!!
#// TODO: ADD ERROR MESSAGES! to all _check() functions
#// TODO: Update memory_check to verify with Internal implementation for the address and other stuff...
#// TODO: verify PC is the same at each step, sequentially! - implemented generally for single cycle
#// TODO: Reorganize code function orders
#// TODO: Support flag args

"""
EXAMPLE:
[F] 01000000 00000093
[D] 01000000 13 01 00 00 0 00 00000000 00
[R] 00 00 00000000 00000000
[E] 01000000 00000000 0
[M] 01000000 00000000 0 0 00000000
[W] 01000000 1 01 00000000

[F] <pc> <instruction>
[D] <pc> <opcode> <rd> <rs1> <rs2> <funct3> <funct7> <imm> <shamt>
[R] <addr_rs1> <addr_rs2> <data_rs1> <data_rs2>
[E] <pc> <alu_result> <branch_taken>
[M] <pc> <memory_addr> <read_write> <access_size> <memory_data>
[W] <pc> <write_enable> <write_rd> <data_rd>
"""

import sys
import argparse
import os

# *** Args Parser
parser = argparse.ArgumentParser()

def trace_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f'{path} is not a file or it does not exist')
    
    if (len(path) <= 6) or (path[-6:] != ".trace"):
        raise argparse.ArgumentTypeError(f'{path} is not a valid trace file (i.e.: *.trace)')

    return path

def hex_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f'{path} is not a file or it does not exist')
    
    if (len(path) <= 2) or (path[-2:] != ".x"):
        raise argparse.ArgumentTypeError(f'{path} is not a valid hex file (i.e.: *.x)')

    return path




# *** Classes
class Memory:
    def __init__(self, size, data_width):
        self.arr = ["0" * data_width] * size
        self.size = size
        self.data_width = data_width

    def set(self, str, index):
        self.arr[index] = str
        return True

    def get(self, index):
        return self.arr[index]
    
    def print(self):
        for i in range(self.size):
            print(f'{i}: {self.arr[i]}')

class DMemory(Memory):
    def __init__(self, size, data_width, mem_path = None):
        super().__init__(size, data_width)

        # Initialize dmemory with contents in mem_path
        if(mem_path):
            # Get the .x file and load its content in binary to dmemory
            with open(mem_path) as f:
                i = 0
                for line in f.readlines():
                    line_bin = dec_to_bin(hex_to_dec(line), 32)
                    if(not self.set_word(line_bin, i)): break
                    i += 4


    def set_byte(self, str, index):
        self.arr[index] = str[-8:]
        return True

    def set_half(self, str, index):
        if(index < self.size): self.arr[index] = str[-8:]
        else: return False
        if(index + 1 < self.size): self.arr[index + 1] = str[-16:-8]
        else: return False
        return True
    
    def set_word(self, str, index):
        if(index < self.size): self.arr[index] = str[-8:]
        else: return False
        if(index + 1 < self.size): self.arr[index + 1] = str[-16:-8]
        else: return False
        if(index + 2 < self.size): self.arr[index + 2] = str[-24:-16]
        else: return False
        if(index + 3 < self.size): self.arr[index + 3] = str[-32:-24]
        else: return False
        # print(f'str: {str}\n{index}: {self.arr[index]}\n{index + 1}: {self.arr[index + 1]}\n{index + 2}: {self.arr[index + 2]}\n{index + 3}: {self.arr[index + 3]}')
        return True

    def get_byte(self, index):
        return self.arr[index]
    
    def get_half(self, index):
        return (
            (self.arr[index + 1] if index + 1 < self.size else "")
            + self.arr[index] 
        )

    def get_word(self, index):
        return (
            (self.arr[index + 3] if index + 3 < self.size else "")
            + (self.arr[index + 2] if index + 2 < self.size else "")
            + (self.arr[index + 1] if index + 1 < self.size else "")
            + self.arr[index] 
        )

class Reg_File(Memory):
    def __init__(self, size, data_width):
        super().__init__(size, data_width)
    
    def set(self, str, index):
        if(index == 0):
            return False
        self.arr[index] = str[-self.size:]
        return True


# *** Flags: Instruction Checks
def is_load_type(instruction):
    return (
        instruction == "LB"
        or instruction == "LH"
        or instruction == "LW"
        or instruction == "LBU"
        or instruction == "LHU"
    )

def is_store_type(instruction):
    return (
        instruction == "SB"
        or instruction == "SH"
        or instruction == "SW"
    )

def is_mem_type(instruction):
    return (
        is_load_type(instruction)
        or is_store_type(instruction)
    )

def is_branch_type(instruction):
    return (
        instruction == "BEQ"
        or instruction == "BNE"
        or instruction == "BLT"
        or instruction == "BGE"
        or instruction == "BLTU"
        or instruction == "BGEU"
    )

def is_jump_type(instruction):
    return (
        instruction == "JAL"
        or instruction == "JALR"
    )

def is_upper_type(instruction):
    return (
        instruction == "LUI"
        or instruction == "AUIPC"
    )

def is_immediate_type(instruction):
    return (
        instruction == "JALR"
        or instruction == "ADDI"
        or instruction == "SLTI"
        or instruction == "SLTIU"
        or instruction == "XORI"
        or instruction == "ORI"
        or instruction == "ANDI"
    )

def is_immediate_shift_type(instruction):
    return (
        instruction == "SLLI"
        or instruction == "SRLI"
        or instruction == "SRAI"
    )

def is_register_type(instruction):
    return (
        instruction == "ADD"
        or instruction == "SUB"
        or instruction == "SLL"
        or instruction == "SLT"
        or instruction == "SLTU"
        or instruction == "XOR"
        or instruction == "SRL"
        or instruction == "SRA"
        or instruction == "OR"
        or instruction == "AND"
    )

def print_instruction(instruction_bin, instr_num = None):
    instruction = decoder(instruction_bin)
    rd = bin_to_dec(get_rd(instruction_bin))
    rs1 = bin_to_dec(get_rs1(instruction_bin))
    rs2 = bin_to_dec(get_rs2(instruction_bin))
    imm = dec_to_hex(bin_to_dec(get_imm(instruction_bin)), 8)
    shamt = dec_to_hex(bin_to_dec(get_shamt(instruction_bin)), 2)

    index_out = f'{instr_num}. ' if instr_num else ""

    if(is_upper_type(instruction) or instruction == "JAL"):
        print(f'{index_out}{instruction} x{rd}, 0x{imm}')

    elif(is_branch_type(instruction)):
        print(f'{index_out}{instruction} x{rs1}, x{rs2}, 0x{imm}')

    elif(is_load_type(instruction)):
        print(f'{index_out}{instruction} x{rd}, 0x{imm}(x{rs1})')

    elif(is_store_type(instruction)):
        print(f'{index_out}{instruction} x{rs2}, 0x{imm}(x{rs1})')
    
    elif(is_immediate_shift_type(instruction)):
        print(f'{index_out}{instruction} x{rd}, x{rs1}, 0x{shamt}')

    elif(is_immediate_type(instruction)):
        print(f'{index_out}{instruction} x{rd}, x{rs1}, 0x{imm}')

    elif(is_register_type(instruction)):
        print(f'{index_out}{instruction} x{rd}, x{rs1}, x{rs2}')

    elif(instruction == "ECALL"):
        print("ECALL")


# *** Getters: Instruction Binary Extraction Functions
def get_trace_args(line):
    return line[4:].split()

def get_opcode(instruction_bin):
    return instruction_bin[-7:]

def get_rd(instruction_bin):
    return instruction_bin[-12:-7]

def get_rs1(instruction_bin):
    return instruction_bin[-20:-15]

def get_rs2(instruction_bin):
    return instruction_bin[-25:-20]

def get_funct3(instruction_bin):
    return instruction_bin[-15:-12]

def get_funct7(instruction_bin):
    return instruction_bin[-32:-25]

def get_imm(instruction_bin):
    imm = ""

    instruction = decoder(instruction_bin)

    # U-TYPE
    if (
        instruction == "LUI"
        or instruction == "AUIPC"
    ):
        imm = zero_extend(instruction_bin[-32:-12], 11)

    # J-TYPES
    elif (instruction == "JAL"):
        imm = sign_extend(zero_extend(instruction_bin[-32] + instruction_bin[-20:-12] + instruction_bin[-21] + instruction_bin[-31:-21], 0), 20)

    # I-TYPES
    elif (
        instruction == "JALR"
        or instruction == "LB"
        or instruction == "LH"
        or instruction == "LW"
        or instruction == "LBU"
        or instruction == "LHU"
        or instruction == "ADDI"
        or instruction == "SLTI"
        or instruction == "SLTIU"
        or instruction == "XORI"
        or instruction == "ORI"
        or instruction == "ANDI"
    ):
        imm = sign_extend(instruction_bin[-32:-20], 11)
    
    # B-TYPES   
    elif(
        instruction == "BEQ"
        or instruction == "BNE"
        or instruction == "BLT"
        or instruction == "BGE"
        or instruction == "BLTU"
        or instruction == "BGEU"
    ):
        temp_string = instruction_bin[-32] + instruction_bin[-8] + instruction_bin[-31:-25] + instruction_bin[-12:-8]
        imm = sign_extend(zero_extend(temp_string, 0), 12)

    # S-TYPES
    elif(
        instruction == "SB"
        or instruction == "SH"
        or instruction == "SW"
    ):
        temp_string = instruction_bin[-32:-25] + instruction_bin[-12:-7]
        imm = sign_extend(temp_string, 11)
        
    # R-TYPES
    elif(
        instruction == "ADD"
        or instruction == "SUB"
        or instruction == "SLL"
        or instruction == "SLT"
        or instruction == "SLTU"
        or instruction == "XOR"
        or instruction == "SRL"
        or instruction == "SRA"
        or instruction == "OR"
        or instruction == "AND"
        or instruction == "SLLI"
        or instruction == "SRLI"
        or instruction == "SRAI"
    ):
        imm = sign_extend("0", 0)

    return imm
    
def get_shamt(instruction_bin):
    return instruction_bin[-25:-20]


# *** Utility Functions: Bitwise Manipulations/Operations and Radix Conversions
# @bin = immediate value
# @sign_bit_index = sign bit index in the imm[]
def sign_extend(bin, sign_bit_index, size = 32):
    return bin[-(sign_bit_index + 1)] * (size - 1 - sign_bit_index) + bin

# Extends down
def zero_extend(bin, start_index):
    return bin + "0" * (start_index + 1)

def twos_complement(bin):
    # Find the first 1 from the right
    index = bin.rfind("1")
    out = bin

    # Flip all above the first 1
    if(index != -1):
        for i in range(index):
            temp = list(out)
            if(out[i] == "0"):
                temp[i] = "1"
            else:
                temp[i] = "0"
            out = "".join(temp)

            # if(out[i] == "0"):
            #     out = out[:i] + "1" + {out[i+1:] if (i+1 < index) else ""}
            # else:
            #     out = out[:i] + "0" + {out[i+1:] if (i+1 < index) else ""}
            
            # out[i] = "1" if out[i] == "0" else "0"
    
    # print(out)
    return out

def signed_less_than(op1, op2):
    # Positive < Negative
    if(int(op1[0]) < int(op2[0])):
        res = False
    # Negative < Positive
    elif(int(op2[0]) < int(op1[0])):
        res = True
    # Same sign
    else:
        res = bin_to_dec(op1) < bin_to_dec(op2)
    return res

def hex_to_dec(hex):
    return int(hex, 16)

def dec_to_hex(dec, size):
    hex = format(dec, f'#0{size + 2}x')
    hex = hex[2:]
    return hex[-size:]

def dec_to_bin(dec, size):
    bin = format(dec, f'#0{size + 2}b')
    bin = bin[2:]
    return bin[-size:]

def bin_to_dec(bin):
    return int(bin, 2)

def signed_bin_to_dec(bin, size = 32):
    return (bin & ((1 << (size-1)) - 1)) - (bin & (1 << (size-1)))



# *** RISC-V PROCESSOR STAGES:
def fetch(line):
    args = get_trace_args(line)
    instruction_hex = args[1]
    pc = args[0]
    return (pc, dec_to_bin(hex_to_dec(instruction_hex), 32))


def decoder(instruction_bin):
    instruction = "N/A"
    opcode = instruction_bin[-7:]
    funct3 = instruction_bin[-15:-12]
    funct7 = instruction_bin[-32:-25]

    # U-TYPES
    if(opcode == "0110111"): instruction = "LUI"
    elif(opcode == "0010111"): instruction = "AUIPC"
    
    # J-TYPE
    elif(opcode == "1101111"): instruction = "JAL"
    
    # B-TYPES
    elif(opcode == "1100011" and funct3 == "000"): instruction = "BEQ"
    elif(opcode == "1100011" and funct3 == "001"): instruction = "BNE"
    elif(opcode == "1100011" and funct3 == "100"): instruction = "BLT"
    elif(opcode == "1100011" and funct3 == "101"): instruction = "BGE"
    elif(opcode == "1100011" and funct3 == "110"): instruction = "BLTU"
    elif(opcode == "1100011" and funct3 == "111"): instruction = "BGEU"

    # I-TYPES
    elif(opcode == "1100111"): instruction = "JALR"
    elif(opcode == "0000011" and funct3 == "000"): instruction = "LB"
    elif(opcode == "0000011" and funct3 == "001"): instruction = "LH"
    elif(opcode == "0000011" and funct3 == "010"): instruction = "LW"
    elif(opcode == "0000011" and funct3 == "100"): instruction = "LBU"
    elif(opcode == "0000011" and funct3 == "101"): instruction = "LHU"
    elif(opcode == "0010011" and funct3 == "000"): instruction = "ADDI"
    elif(opcode == "0010011" and funct3 == "010"): instruction = "SLTI"
    elif(opcode == "0010011" and funct3 == "011"): instruction = "SLTIU"
    elif(opcode == "0010011" and funct3 == "100"): instruction = "XORI"
    elif(opcode == "0010011" and funct3 == "110"): instruction = "ORI"
    elif(opcode == "0010011" and funct3 == "111"): instruction = "ANDI"
    elif(opcode == "0010011" and funct3 == "001"): instruction = "SLLI"
    elif(opcode == "0010011" and funct3 == "101" and funct7 == "0000000"): instruction = "SRLI"
    elif(opcode == "0010011" and funct3 == "101" and funct7 == "0100000"): instruction = "SRAI"

    # S-TYPES
    elif(opcode == "0100011" and funct3 == "000"): instruction = "SB"
    elif(opcode == "0100011" and funct3 == "001"): instruction = "SH"
    elif(opcode == "0100011" and funct3 == "010"): instruction = "SW"

    # R-TYPES
    elif(opcode == "0110011" and funct3 == "000" and funct7 == "0000000"): instruction = "ADD"
    elif(opcode == "0110011" and funct3 == "000" and funct7 == "0100000"): instruction = "SUB"
    elif(opcode == "0110011" and funct3 == "001"): instruction = "SLL"
    elif(opcode == "0110011" and funct3 == "010"): instruction = "SLT"
    elif(opcode == "0110011" and funct3 == "011"): instruction = "SLTU"
    elif(opcode == "0110011" and funct3 == "100"): instruction = "XOR"
    elif(opcode == "0110011" and funct3 == "101" and funct7 == "0000000"): instruction = "SRL"
    elif(opcode == "0110011" and funct3 == "101" and funct7 == "0100000"): instruction = "SRA"
    elif(opcode == "0110011" and funct3 == "110"): instruction = "OR"
    elif(opcode == "0110011" and funct3 == "111"): instruction = "AND"
    
    elif(opcode == "1110011" and funct3 == "000" and funct7 == "0000000"): instruction = "ECALL"

    else: instruction = "N/A"
    
    return instruction

def decoder_check(line, instruction_bin, pc):
    args = get_trace_args(line)
    instruction = decoder(instruction_bin)

    # External: In trace file
    opcode = dec_to_bin(hex_to_dec(args[1]), 7)
    rd = dec_to_bin(hex_to_dec(args[2]), 5)
    rs1 = dec_to_bin(hex_to_dec(args[3]), 5)
    rs2 = dec_to_bin(hex_to_dec(args[4]), 5)
    funct3 = dec_to_bin(hex_to_dec(args[5]), 3)
    funct7 = dec_to_bin(hex_to_dec(args[6]), 7)
    imm = dec_to_bin(hex_to_dec(args[7]), 32)
    shamt = dec_to_bin(hex_to_dec(args[8]), 5)

    # Internal: In processor (expected values)
    exp_opcode = get_opcode(instruction_bin)
    exp_rd = get_rd(instruction_bin)
    exp_rs1 = get_rs1(instruction_bin)
    exp_rs2 = get_rs2(instruction_bin)
    exp_funct3 = get_funct3(instruction_bin)
    exp_funct7 = get_funct7(instruction_bin)
    exp_imm = get_imm(instruction_bin)
    exp_shamt = get_shamt(instruction_bin)

    res = True
    err = None

    # U-TYPES & J-TYPE
    if (
        (
            instruction == "LUI"
            or instruction == "AUIPC"
            or instruction == "JAL"
        )
        and not
        (
            opcode == exp_opcode
            and rd == exp_rd
            and imm == exp_imm
        )
    ):        
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nrd: Got: {rd}, Expected: {exp_rd}.'
            f'\nimm: Got: {imm}, Expected: {exp_imm}.'
        )
        
    # I-TYPES
    elif (
        (
            instruction == "JALR"
            or instruction == "LB"
            or instruction == "LH"
            or instruction == "LW"
            or instruction == "LBU"
            or instruction == "LHU"
            or instruction == "ADDI"
            or instruction == "SLTI"
            or instruction == "SLTIU"
            or instruction == "XORI"
            or instruction == "ORI"
            or instruction == "ANDI"
        )
        and not
        (
            opcode == exp_opcode
            and rd == exp_rd
            and funct3 == exp_funct3
            and rs1 == exp_rs1
            and imm == exp_imm
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nrd: Got: {rd}, Expected: {exp_rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {exp_funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {exp_rs1}.'
            f'\nimm: Got: {imm}, Expected: {exp_imm}.'
        )
    
    # B-TYPES   
    elif(
        (
            instruction == "BEQ"
            or instruction == "BNE"
            or instruction == "BLT"
            or instruction == "BGE"
            or instruction == "BLTU"
            or instruction == "BGEU"
        )
        and not
        (
            opcode == exp_opcode
            and imm == exp_imm
            and funct3 == exp_funct3
            and rs1 == exp_rs1
            and rs2 == exp_rs2
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nimm: Got: {imm}, Expected: {exp_imm}.'
            f'\nfunct3: Got: {funct3}, Expected: {exp_funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {exp_rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {exp_rs2}.'
        )

    # S-TYPES
    elif(
        (
            instruction == "SB"
            or instruction == "SH"
            or instruction == "SW"
        )
        and not
        (
            opcode == exp_opcode
            and imm == exp_imm
            and rs1 == exp_rs1
            and rs2 == exp_rs2
            and funct3 == exp_funct3
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nimm: Got: {imm}, Expected: {exp_imm}.'
            f'\nfunct3: Got: {funct3}, Expected: {exp_funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {exp_rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {exp_rs2}.'
        )
        
    # R-TYPES with SHAMT
    elif(
        (
            instruction == "SLLI"
            or instruction == "SRLI"
            or instruction == "SRAI"
        )
        and not
        (
            opcode == exp_opcode
            and rd == exp_rd
            and funct3 == exp_funct3
            and rs1 == exp_rs1
            and shamt == exp_shamt
            and funct7 == exp_funct7
        )
    ):
        # print(instruction_bin)
        # print(opcode == exp_opcode)
        # print(rd == exp_rd)
        # print(funct3 == exp_funct3)
        # print(rs1 == exp_rs1)
        # print(shamt == exp_shamt)
        # print(funct7 == exp_funct7)
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nrd: Got: {rd}, Expected: {exp_rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {exp_funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {exp_rs1}.'
            f'\nshamt: Got: {shamt}, Expected: {exp_shamt}.'
            f'\nfunct7: Got: {funct7}, Expected: {exp_funct7}.'
        )

    # R-TYPES
    elif(
        (
            instruction == "ADD"
            or instruction == "SUB"
            or instruction == "SLL"
            or instruction == "SLT"
            or instruction == "SLTU"
            or instruction == "XOR"
            or instruction == "SRL"
            or instruction == "SRA"
            or instruction == "OR"
            or instruction == "AND"
        )
        and not
        (
            opcode == exp_opcode
            and rd == exp_rd
            and funct3 == exp_funct3
            and rs1 == exp_rs1
            and rs2 == exp_rs2
            and funct7 == exp_funct7
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nrd: Got: {rd}, Expected: {exp_rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {exp_funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {exp_rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {exp_rs2}.'
            f'\nfunct7: Got: {funct7}, Expected: {exp_funct7}.'
        )
    
    elif (
        instruction == "ECALL"
        and not 
        (
            opcode == exp_opcode
            and instruction_bin[-32:-7] == "0" * 25
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - opcode: Got: {opcode}, Expected: {exp_opcode}.'
            f'\nOther upper bits: Got: {instruction_bin[-32:-7]}, Expected: {"0" * 25}.'
        )
    
    return (res, err)


def execute(instruction_bin, pc, reg_file):
    instruction = decoder(instruction_bin)
    rs1 = reg_file.get(bin_to_dec(get_rs1(instruction_bin)))
    rs2 = reg_file.get(bin_to_dec(get_rs2(instruction_bin)))
    imm = get_imm(instruction_bin)
    shamt = get_shamt(instruction_bin)
    
    alu_res = "0" * 32
    br_taken = "0"

    # J-Types - take the branch (pc = ALU output)
    if(instruction == "JAL" or instruction == "JALR"):
        br_taken = "1"


    # U-TYPES
    if (instruction == "LUI"): alu_res = zero_extend(imm[-32:-12], 11)
    
    # Add IMM to PC
    elif (
        instruction == "AUIPC"
        or instruction == "JAL"
    ): 
        alu_res = dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32)
        
    # B-TYPES
    elif(
        instruction == "BEQ"
        or instruction == "BNE"
        or instruction == "BLT"
        or instruction == "BGE"
        or instruction == "BLTU"
        or instruction == "BGEU"
    ):
        if (instruction == "BEQ"): br_taken = "1" if rs1 == rs2 else "0"
        elif (instruction == "BNE"): br_taken = "1" if not(rs1 == rs2) else "0"
        elif (instruction == "BLT"): br_taken = "1" if signed_less_than(rs1, rs2) else "0"
        elif (instruction == "BGE"): br_taken = "1" if not(signed_less_than(rs1, rs2)) else "0"
        elif (instruction == "BLTU"): br_taken = "1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0"
        elif (instruction == "BGEU"): br_taken = "1" if not(bin_to_dec(rs1) < bin_to_dec(rs2)) else "0"

        # Always calculate ALU result for branch
        alu_res = dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32)

    # I-TYPES
    elif (
        instruction == "JALR"
        or instruction == "LB"
        or instruction == "LW"
        or instruction == "LH"
        or instruction == "LBU"
        or instruction == "LHU"
        or instruction == "SB"
        or instruction == "SH"
        or instruction == "SW"
        or instruction == "ADDI"
    ):
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32)

    elif (instruction == "SLTI"):
        alu_res = "0" * 31 + ("1" if signed_less_than(rs1, imm) else "0")

    elif (instruction == "SLTIU"):
        alu_res = "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(imm) else "0")

    elif (instruction == "XORI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(imm), 32)

    elif (instruction == "ORI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) | bin_to_dec(imm), 32)

    elif (instruction == "ANDI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) & bin_to_dec(imm), 32)

    elif (instruction == "SLLI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) << bin_to_dec(shamt), 32)

    elif (instruction == "SRLI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 32)

    elif (instruction == "SRAI"):
        alu_res = rs1[0] * bin_to_dec(shamt) + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 32 - bin_to_dec(shamt))


    # R-TYPES
    elif (instruction == "ADD"):
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(rs2), 32)

    elif (instruction == "SUB"):
        # Two's complement method of adding
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(twos_complement(rs2)), 32)

    elif (instruction == "SLL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) << bin_to_dec(rs2[-5:]), 32)

    elif (instruction == "SLT"):
        alu_res = "0" * 31 + ("1" if signed_less_than(rs1, rs2) else "0")

    elif (instruction == "SLTU"):
        alu_res = "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0")

    elif (instruction == "XOR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(rs2), 32)

    elif (instruction == "SRL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2[-5:]), 32)

    elif (instruction == "SRA"):
        alu_res = rs1[0] * bin_to_dec(rs2[-5:]) + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2[-5:]), 32 - bin_to_dec(rs2[-5:]))

    elif (instruction == "OR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) | bin_to_dec(rs2), 32)

    elif (instruction == "AND"):
        alu_res = dec_to_bin(bin_to_dec(rs1) & bin_to_dec(rs2), 32)

    # elif (instruction == "ECALL"):

    return (alu_res, br_taken)

def execute_check(line, instruction_bin, reg_file):
    res = True
    err = None

    instruction = decoder(instruction_bin)

    # External: In trace file
    exec_args = get_trace_args(line)
    pc = dec_to_bin(hex_to_dec(exec_args[0]), 32)
    alu_res = dec_to_bin(hex_to_dec(exec_args[1]), 32)
    br_taken = dec_to_bin(hex_to_dec(exec_args[2]), 1)

    # Internal: In processor
    (exp_alu_res, exp_br_taken) = execute(instruction_bin, pc, reg_file)
    
    # B-TYPES
    if (
        (
            instruction == "BEQ"
            or instruction == "BNE"
            or instruction == "BLT"
            or instruction == "BGE"
            or instruction == "BLTU"
            or instruction == "BGEU"
        )
        and not
        (
            br_taken == exp_br_taken
            and
            (
                (
                    br_taken == "1"
                    and alu_res == exp_alu_res
                )
                or br_taken == "0" 
            )
        )
    ):
        res = False
        err = (
            f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - branch_taken: {br_taken}, expected: {exp_br_taken}.'
            + f'\n{instruction}: alu_result: {alu_res}, expected (if branch taken): {exp_alu_res}'
        )
    
    # Other Types
    # ???: branch_taken is a DC for jumps/J-Types?
    elif (
        (
            instruction == "LUI"
            or instruction == "AUIPC"
            or instruction == "JAL" 
            or instruction == "JALR"
            or instruction == "LB"
            or instruction == "LW"
            or instruction == "LH"
            or instruction == "LBU"
            or instruction == "LHU"
            or instruction == "SB"
            or instruction == "SH"
            or instruction == "SW"
            or instruction == "ADDI"
            or instruction == "SLTI"
            or instruction == "SLTIU"
            or instruction == "XORI"
            or instruction == "ORI"
            or instruction == "ANDI"
            or instruction == "SLLI"
            or instruction == "SRLI"
            or instruction == "SRAI"
            or instruction == "ADD"
            or instruction == "SUB"
            or instruction == "SLL"
            or instruction == "SLT"
            or instruction == "SLTU"
            or instruction == "XOR"
            or instruction == "SRL"
            or instruction == "SRA"
            or instruction == "OR"
            or instruction == "AND"
        )
        and alu_res != exp_alu_res
    ): 
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - alu_result: {alu_res}, expected: {exp_alu_res}'

    elif (instruction == "ECALL"):
        # ECALL Warning!
        res = True
        if(instruction_bin[-32:-7] != "0" * (25)):
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - WARNING - upper 25 bits are not all 0s: got: {instruction_bin[-32:-7]}, expected: {"0" * (25)}'

    return (res, err)


def register_check(line, instruction_bin, reg_file):
    res = True
    err = None

    # External: In trace
    args = get_trace_args(line)
    addr_rs1 = hex_to_dec(args[0])
    addr_rs2 = hex_to_dec(args[1])
    rs1 = args[2] # in hex
    rs2 = args[3] # in hex

    # Internal: In processor
    instruction = decoder(instruction_bin)
    exp_addr_rs1 = bin_to_dec(get_rs1(instruction_bin))
    exp_addr_rs2 = bin_to_dec(get_rs2(instruction_bin))
    exp_rs1 = dec_to_hex(bin_to_dec(reg_file.get(exp_addr_rs1)), 8)
    exp_rs2 = dec_to_hex(bin_to_dec(reg_file.get(exp_addr_rs2)), 8)
    
    # Verify addresses
    # For instructions utilizing rs1 and/or rs2
    if(
        not (
            is_upper_type(instruction)
            or instruction == "JAL"
        )
        and
        addr_rs1 != exp_addr_rs1
    ):
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - Incorrect rs1 address. Got: {addr_rs1}. Expected: {exp_addr_rs1}'
    elif(
        not (
            is_upper_type(instruction)
            or instruction == "JAL"
            or is_load_type(instruction)
            or is_immediate_type(instruction)
            or is_immediate_shift_type(instruction)
        )
        and
        addr_rs2 != exp_addr_rs2
    ):
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - Incorrect rs2 address. Got: {addr_rs2}. Expected: {exp_addr_rs2}'

    # Verify register values
    elif(
        not (
            is_upper_type(instruction)
            or instruction == "JAL"
        )
        and
        rs1 != exp_rs1
    ):
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - Incorrect rs1 data at address {addr_rs1}. Got: {rs1}. Expected: {exp_rs1}.'
    elif(
        not (
            is_upper_type(instruction)
            or instruction == "JAL"
            or is_load_type(instruction)
            or is_immediate_type(instruction)
            or is_immediate_shift_type(instruction)
        )
        and
        rs2 != exp_rs2
    ):
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - Incorrect rs2 data at address {addr_rs2}. Got: {rs2}. Expected: {exp_rs2}.'

    return (res, err)


# Executes stores, loads can be access later during memory checking
def memory(dmemory, instruction, addr, write_data):
    out = None

    mem_index = bin_to_dec(addr[-12:])
    # mem_index = bin_to_dec(addr)
    # print(f'MEMORY: write_data: {write_data}')

    if(instruction == "SB"):
        out = write_data[-8:]
        dmemory.set_byte(write_data, mem_index)
    elif(instruction == "SH"):
        out = write_data[-16:]
        dmemory.set_half(write_data, mem_index)
    elif(instruction == "SW"):
        out = write_data[-32:]
        dmemory.set_word(write_data, mem_index)
    elif(instruction == "LB"): out = sign_extend(dmemory.get_byte(mem_index), 7)
    elif(instruction == "LH"): out = sign_extend(dmemory.get_half(mem_index), 15)
    elif(instruction == "LW"): out = dmemory.get_word(mem_index)[-32:]
    elif(instruction == "LBU"): out = "0" * 24 + dmemory.get_byte(mem_index)[-8:]
    elif(instruction == "LHU"): out = "0" * 16 + dmemory.get_half(mem_index)[-16:]

    return out

def memory_check(line, dmemory, instruction_bin, addr, write_data):
    args = get_trace_args(line)
    instruction = decoder(instruction_bin)
    res = True
    err = None
    
    # External: In trace file
    mem_addr = dec_to_bin(hex_to_dec(args[1]), 32)
    rw = args[2] == "1"
    access_size = args[3]
    data = dec_to_bin(hex_to_dec(args[4]), 32)


    # Verify access_size, address, read_write, and memory_data!
    # access_size & address:
    if (is_mem_type(instruction)):
        if(
            (
                instruction == "LB" 
                or instruction == "LBU"
                or instruction == "SB"
            )
            and access_size != "0"
        ):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - access_size is {access_size}, expecting 0'

        elif(
            (
                instruction == "LH" 
                or instruction == "LHU"
                or instruction == "SH"
            )
            and access_size != "1"
        ):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - access_size is {access_size}, expecting 1'

        elif(
            (
                instruction == "LW" 
                or instruction == "SW"
            )
            and access_size != "2"
        ):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - access_size is {access_size}, expecting 2'
        
        # address
        elif(mem_addr != addr):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - trace address is not the same as memory address. addr: Got: {mem_addr}, Expected: {addr}'

    # read_write:
    # Default - Read: rw = 0. On stores - Write: rw = 1
    # If it is not a store instruction, it should not be writing
    if(res):
        if(
            rw
            and
            not (
                instruction == "SB"
                or instruction == "SH"
                or instruction == "SW"
            )
        ):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - should NOT be writing to dmemory if not a Store instruction (SB, SH, SW only)'


    # memory_data:
    if(res):
        addr_dec = bin_to_dec(addr[-12:])
        # addr_dec = bin_to_dec(addr)
        dmem_data = dmemory.get_word(addr_dec)

        # Check if state of dmemory is correct on reads
        # Only If it is a memory instruction. DC otherwise
        if(is_load_type(instruction)):
            if(
                (
                    instruction == "LB" 
                    or instruction == "LBU"
                )  
                and data[-8:] != dmem_data[-8:]
            ):
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory byte at address {addr_dec} is {dmem_data[-8:]}, got {data[-8:]}'

            elif(
                (
                    instruction == "LH"
                    or instruction == "LHU"
                )  
                and data[-16:] != dmem_data[-16:]
            ):
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory half-word at address {addr_dec} is {dmem_data[-16:]}, got {data[-16:]}'


            elif(
                (
                    instruction == "LW"
                )
                and data[-32:] != dmem_data[-32:]
            ): 
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory word at address {addr_dec} is {dmem_data[-32:]}, got {data[-32:]}'
        
        # Verify that data was stored correctly
        elif(is_store_type(instruction)):
            if(
                instruction == "SB"  
                and dmem_data[-8:] != write_data[-8:]
            ):
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory byte at address {addr_dec} is {dmem_data[-8:]}, expecting {write_data[-8:]}'

            elif(
                instruction == "SH"  
                and dmem_data[-16:] != write_data[-16:]
            ):
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory half-word at address {addr_dec} is {dmem_data[-16:]}, expecting {write_data[-16:]}'


            elif(
                instruction == "SW"
                and dmem_data[-32:] != write_data[-32:]
            ): 
                res = False
                err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - dmemory word at address {addr_dec} is {dmem_data[-32:]}, expecting {write_data[-32:]}'

    return (res, err)


def write(instruction_bin, alu_res, mem_res, pc, reg_file):
    instruction = decoder(instruction_bin)
    # opcode = get_opcode(instruction_bin)
    rd_addr = bin_to_dec(get_rd(instruction_bin))
    # rd = reg_file.get(bin_to_dec(get_rd(instruction_bin)))
    # rs1 = reg_file.get(bin_to_dec(get_rs1(instruction_bin)))
    # rs2 = reg_file.get(bin_to_dec(get_rs2(instruction_bin)))
    # funct3 = get_funct3(instruction_bin)
    # funct7 = get_funct7(instruction_bin)
    # imm = get_imm(instruction_bin)
    # shamt = get_shamt(instruction_bin)

    out = None
    res = True

    # WB: PC+4
    if(
        instruction == "JAL"
        or instruction == "JALR"
    ):
        out = dec_to_bin(hex_to_dec(pc) + 4, 32)

    # WB: MEM
    elif (
        instruction == "LB"
        or instruction == "LH"
        or instruction == "LW"
        or instruction == "LBU"
        or instruction == "LHU"
    ): 
        out = mem_res

    # WB: ALU
    elif (
        instruction == "LUI"
        or instruction == "AUIPC"
        or instruction == "ADDI"
        or instruction == "SLTI"
        or instruction == "SLTIU"
        or instruction == "XORI"
        or instruction == "ORI"
        or instruction == "ANDI"
        or instruction == "SLLI"
        or instruction == "SRLI"
        or instruction == "SRAI"
        or instruction == "ADD"
        or instruction == "SUB"
        or instruction == "SLL"
        or instruction == "SLT"
        or instruction == "SLTU"
        or instruction == "XOR"
        or instruction == "SRL"
        or instruction == "SRA"
        or instruction == "OR"
        or instruction == "AND"
    ):
    # else:
       out = alu_res

    if(out):
        # print(f'WRITE: x{rd_addr}: {out}')
        res = reg_file.set(out, rd_addr)
    # res = reg_file.set(out, rd_addr)

    return (res, out)
    
def write_check(line, instruction_bin, write_res, reg_file):
    res = True
    err = None

    # In Trace
    args = get_trace_args(line)
    pc = args[0]
    write_enable = args[1] == "1"
    write_rd = hex_to_dec(args[2])
    data_rd = dec_to_bin(hex_to_dec(args[3]), 32)

    # In RISC-V Processor
    instruction = decoder(instruction_bin)
    rd_addr = bin_to_dec(get_rd(instruction_bin))
    rd = reg_file.get(rd_addr)
    exp_write = write_res[1]

    # Verify write_enable
    if(
        write_enable
        and (
            is_branch_type(instruction)
            or is_store_type(instruction)
            or instruction == "ECALL"
        )
    ):
        res = False
        err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - should NOT be writing to destination register, rd; (register) write_enable is "1", expecting "0"'
    elif(
        not (
            is_branch_type(instruction)
            or is_store_type(instruction)
            or instruction == "ECALL"
        )
    ):
        if(not write_enable):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - should be writing to destination register, rd; (register) write_enable is "0", expecting "1"'

        elif(write_rd != rd_addr):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - DECODER - incorrect destination register address, rd; got {write_rd}, expecting {rd_addr}'

        elif(data_rd != exp_write):
            res = False
            err = f'<{instruction}>: {dec_to_hex(bin_to_dec(instruction_bin), 8)} - DECODER - incorrect write back data; got {data_rd}, expecting {exp_write}'
    
    return (res, err)



def main():
    parser.add_argument("trace", type=trace_file, help="Path to the a .trace file (ex. ../sim/verilator/test_pd/rv32ui-p-sltiu.trace)")
    parser.add_argument("hex", type=hex_file, help="Path to the a .x file (ex. /rv32-benchmarks/individual-instructions/rv32ui-p-sltiu.x). Used to initialize dmemory")
    parser.add_argument("-i", "--instructions", dest="instructions", action="store_true", help="Print all instructions in RISC-V format (Default: False, instructions enumerated)")
    parser.add_argument("-in", "--instructions-no-num", dest="instructions_no_num", action="store_true", help="Print all instructions in RISC-V format (Default: False, instructions NOT enumerated)")
    parser.add_argument("-r", "--regfile", dest="regfile", action="store_true", help="Print the state of the Register File at each [R] (Default: False)")
    parser.add_argument("-t", "--terminate", "--ecall", dest="ecall", action="store_true", help="End (terminate) the parse.py check at the first ECALL instruction (Default: False)")
    parser.add_argument("-s", "--skip", dest="skip", action="store_false", help="CONTINUE execution on errors (Default: True)")
    parser.add_argument("-m", "--mem", dest="mem", type=int, default=1048576, metavar="MEM_DEPTH", help="Number of bytes (8-bit values) in data memory (dmemory). (Default: 1048576)")
    args = parser.parse_args()

    TRACE = args.trace
    HEX_FILE = args.hex
    SHOW_INSTR = args.instructions
    SHOW_INSTR_NO_NUM = args.instructions_no_num
    SHOW_REG = args.regfile
    END_ON_ECALL = args.ecall
    STOP_ON_ERR = args.skip
    MEM_DEPTH = args.mem # # of bytes (8-bit values)

    if MEM_DEPTH <= 0:
        print(f'Please enter a MEM_DEPTH greater than 0. Got: {MEM_DEPTH}\n')
        return

    f = open(TRACE, 'r')
    line = f.readline()

    reg_file = Reg_File(32, 32)
    dmemory = DMemory(MEM_DEPTH, 8, HEX_FILE) # Byte-addressable memory
    instruction_bin = ""
    instruction = ""
    pc = ""
    current_pc = ""
    # exp_pc = "" # for multi-cycle processor
    dec_args = []
    exec_res = (None, None)
    alu_res = None
    mem_res = None

    # Internal: From decoder
    rd = None
    rs1 = None
    rs2 = None
    imm = None


    iterator = 0
    instr_num = 0

    while (line):
        iterator += 1
        # print(line)

        # PC CHECK!
        if(
            (
                line[1] == 'D'
                or line[1] == 'E'
                or line[1] == 'M'
                or line[1] == 'W'
            )
            and iterator > 1
        ):
            curr_args = get_trace_args(line)
            pc = curr_args[0]

            # FOR SINGLE CYCLE PROCESSOR!
            if(pc != current_pc):
                print(
                    f'<{line[1]}> Incorrect PC. Got: {pc}. Expected: {current_pc}'
                    f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n'
                )
                if(STOP_ON_ERR):
                    break

        

        if(line[1] == 'F'):
            (pc, instruction_bin) = fetch(line)
            # print(f'INSTR BINARY:{instruction_bin}')
            current_pc = pc
                

        elif(line[1] == 'D'):
            if(instruction_bin != "0" * 32):
                instruction = decoder(instruction_bin)
                rd = get_rd(instruction_bin)
                rs1 = get_rs1(instruction_bin)
                rs2 = get_rs2(instruction_bin)
                imm = get_imm(instruction_bin)

                instr_num += 1
                if(SHOW_INSTR or SHOW_INSTR_NO_NUM): 
                    print_instruction(instruction_bin, instr_num if SHOW_INSTR else None)

                (success, err) = decoder_check(line, instruction_bin, current_pc)
                if(not success and instruction != "N/A"):
                    print(
                        f'<DECODER> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, did not DECODE correctly.'
                    )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success and instruction != "N/A"):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break
                if(END_ON_ECALL and instruction == "ECALL"):
                     print(f'\nECALL: $finish (end of program)\n')
                     break

                        
        elif(line[1] == 'R'):
            if(instruction_bin != "0" * 32 and instruction != "N/A"):
                (success, err) = register_check(line, instruction_bin, reg_file)
                if(SHOW_REG): reg_file.print()

                if(not success):
                    print(
                        f'<REGISTER> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, REGISTER FILE was not updated correctly.'
                    )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break


        elif(line[1] == 'E'):
            if(instruction_bin != "0" * 32 and instruction != "N/A"):
                exec_res = execute(instruction_bin, dec_to_bin(hex_to_dec(pc), 32), reg_file)
                alu_res = exec_res[0]
                (success, err) = execute_check(line, instruction_bin, reg_file)
                if (not success):
                        print(
                            f'<EXECUTE> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, did not EXECUTE correctly.'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break

                # PC CHECK!
                # exec_args = get_trace_args(line)
                # pc = exec_args[0]
                

        elif(line[1] == 'M'):
            if(instruction_bin != "0" * 32 and instruction != "N/A"):
                # memory at address alu_res, write data = value at rs2 (for store instructions)
                mem_res = memory(dmemory, instruction, alu_res, reg_file.get(bin_to_dec(rs2)))
                (success, err) = memory_check(line, dmemory, instruction_bin, alu_res, reg_file.get(bin_to_dec(rs2)))
                if (not success):
                        print(
                            f'<MEMORY> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, MEMORY (dmemory) was not updated correctly.'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break
                
                # PC CHECK!
                # mem_args = get_trace_args(line)
                # pc = mem_args[0]


        elif(line[1] == 'W'):
            if(instruction_bin != "0" * 32 and instruction != "N/A"):
                # memory at address rd, write data = value at rs1
                write_res = write(instruction_bin, alu_res, mem_res, current_pc, reg_file)
                (success, err) = write_check(line, instruction_bin, write_res, reg_file)
                if (not success):
                        print(
                            f'<WRITE> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, did not WRITE correctly.'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break

                # PC CHECK!
                # write_args = get_trace_args(line)
                # pc = write_args[0]
    
        line = f.readline()

    f.close()


if __name__ == "__main__":
    main()
