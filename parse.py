'''
    Authors: Justin Mendes and Shazil Razzaq
    Date: Monday October 17, 2022
    Last Modified: Friday Dec 2, 2022
    Version: v1.4

    Tests functionality of RISC-V processor up to PD4

    USAGE:  python3 parse.py [-h/--help] [-s/--skip] [-i/--instructions] [-in/--instructions-no-num] 
            [-r/--regfile] [-m <MEM_DEPTH> / --mem <MEM_DEPTH>] 
            <PATH to .trace file> <PATH to .x file>

    EX:     python3 parse.py rv32ui-p-sltiu.trace /rv32-benchmarks/individual-instructions/rv32ui-p-sltiu.x

    UPDATES:
    @Shazil-R && @justincmendes:
    - up to PD5 instead of up to PD4
    - refactored code: more pythonic, more OOP

    FUTURE:
    v1.x: 
        - Add support for flags (-pd1 -pd2 -pd3 -pd4 -pd5)
            To separate the parsers verification for each of the different projects
            (upto and including the project)
'''

#// TODO: ADD REGISTER FILE + DMEMORY + WRITEBACK LOGIC!!!
#// TODO: ADD ERROR MESSAGES! to all _check() functions
#// TODO: Update memory_check to verify with Internal implementation for the address and other stuff...
#// TODO: verify PC is the same at each step, sequentially! - implemented generally for single cycle
#// TODO: Reorganize code function orders
#// TODO: Support flag args

'''
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
'''

import sys
import argparse
import os
import glob
from collections import namedtuple

SP_BASE = "01000000" # in hex
Res = namedtuple('Result', ['err', 'res'])

'''Args Parser'''
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



'''Classes'''
#// TODO: Instruction Class: define this with all of the properties and control signals associated with any instruction at any time in the pipeline
class Instruction():
    #// TODO: __init__ as NOP/KILL, no params
    def __init__(self, id=-1):
        self.id = id
        self.name = "NOP"
        
        self.pc = "0" * 32
        self.binary = "0" * 32
        self.opcode = "0" * 7
        self.rd = "0" * 5
        self.rs1 = "0" * 5
        self.rs2 = "0" * 5
        self.funct3 = "0" * 3 
        self.funct7 = "0" * 7
        self.imm = "0" * 32
        self.shamt = "0" * 5
        
        self.pc_sel = "0" 
        self.br_eq = "0"
        self.br_lt = "0"
        self.br_un = "0"
        self.mem_rw = "0"
        self.reg_w_en = "0"
        self.wb_sel = "0" * 2
        self.rs1_data = "0" * 32
        self.rs2_data = "0" * 32
        self.alu_result = "0" * 32
        self.memory_access_size = "0" * 2
        self.memory_data = "0" * 32
        self.write_data = "0" * 32
    
    def print(self):
        print(f'id: {self.id}')
        print(f'name: {self.name}')
        print(f'pc: {dec_to_hex(bin_to_dec(self.pc), 8)} ({self.pc})')
        print(f'binary: {self.binary}')
        print(f'opcode: {self.opcode}')
        print(f'rd: {self.rd}')
        print(f'rs1: {self.rs1}')
        print(f'rs2: {self.rs2}')
        print(f'funct3: {self.funct3}')
        print(f'funct7: {self.funct7}')
        print(f'imm: {self.imm}')
        print(f'shamt: {self.shamt}')

        print(f'pc_sel: {self.pc_sel}')
        print(f'br_eq: {self.br_eq}')
        print(f'br_lt: {self.br_lt}')
        print(f'br_un: {self.br_un}')
        print(f'mem_rw: {self.mem_rw}')
        print(f'reg_w_en: {self.reg_w_en}')
        print(f'wb_sel: {self.wb_sel}')
        print(f'rs1_data: {self.rs1_data}')
        print(f'rs2_data: {self.rs2_data}')
        print(f'alu_result: {self.alu_result}')
        print(f'memory_access_size: {self.memory_access_size}')
        print(f'memory_data: {self.memory_data}')
        print(f'write_data: {self.write_data}')

        
#// TODO: Pipeline Class
class Pipeline():
    def __init__(self): 
        self.size = 5
        
        self.queue = [-1] * self.size # 5-element array (for 5-stages)
        for i in range(self.size):
            self.queue[i] = Instruction() 
        
        # self.queue = [Instruction()] * self.size # 5-element array (for 5-stages)

        # blank_fetch = Instruction("Fetch")
        # blank_decode = Instruction("Decode")
        # blank_execute = Instruction("Execute")
        # blank_memory = Instruction("Memory")
        # blank_write = Instruction("Write")
        # self.queue = [-1] * self.size # 5-element array (for 5-stages)
        # self.queue = [blank_fetch, blank_decode, blank_execute, blank_memory, blank_write]
        # self.queue.append(blank_fetch)
        # self.queue.append(blank_decode)
        # self.queue.append(blank_execute)
        # self.queue.append(blank_memory)
        # self.queue.append(blank_write)

        # self.queue = [Instruction()] * self.size # 5-element array (for 5-stages)
        
        # self.queue = [Instruction() for i in range(5)]

        # self.queue = [-1] * 5 # 5-element array (for 5-stages)
        # for i in range(5):
        #     blank_instruction = Instruction()
        #     self.queue[i] = blank_instruction

        # self.queue = []
        # for i in range(5):
        #     self.queue.append(Instruction())

        # print(blank_fetch, blank_decode, blank_execute, blank_memory, blank_write)
        
        # print("===== INIT =====")
        # self.print()

    def print(self):
        for i in range(5):
            stage = ""
            if(i == 0): stage = "Fetch:"
            elif(i == 1): stage = "Decode:"
            elif(i == 2): stage = "Execute:"
            elif(i == 3): stage = "Memory:"
            elif(i == 4): stage = "Write:"

            # print (f'{stage} 0x{dec_to_hex(bin_to_dec(self.queue[i].pc), 8)} {self.queue[i].name} {self.queue[i].binary} rd:{self.queue[i].rd} rs1:{self.queue[i].rs1} rs2:{self.queue[i].rs2} imm:{self.queue[i].imm} (ID: {self.queue[i].id})')
            print (f'{stage} 0x{dec_to_hex(bin_to_dec(self.queue[i].pc), 8)} {get_print_instruction(self.queue[i])} (ID: {self.queue[i].id})')


    def add(self, instruction=Instruction()):
        '''Right-shift then (enqueue) push to front of FIFO queue'''
        # Right shift from beginning (Fetch onwards)
        self.queue = [instruction] + self.queue[:-1]

    def stall(self):
        '''Injects NOP into EXECUTE stage'''
        nop = Instruction()
        # Right shift from Execute onwards 
        self.queue = self.queue[0:2] + [nop] + self.queue[2:-1]

    def flush_jump(self):
        '''Kill Decode instruction once a J-Type is resolved'''
        nop = Instruction(self.queue[1].id)
        self.queue[1] = nop

    def flush_branch(self):
        '''
            Kill Decode and Execute instructions if branch should be taken
            Default: Branch NOT taken (static)
        '''
        for i in range(1, 3):
            nop = Instruction(self.queue[i].id)
            self.queue[i] = nop



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


class RegFile(Memory):
    def __init__(self, size, data_width):
        super().__init__(size, data_width)
    
    def set(self, str, index):
        if(index == 0):
            return False
        self.arr[index] = str[-self.size:]
        return True

    # !comment this out after debugging
    # def print(self):
    #     print(f'x0: {self.arr[0]}')
    #     print(f'x1: {self.arr[1]}')
    #     print(f'x2: {self.arr[2]}\n')


'''Flags: Instruction Checks'''
def is_reg_writeback_type(instruction_name):
    return (
        not is_store_type(instruction_name) and not is_branch_type(instruction_name) 
        and instruction_name != "ECALL"
        and instruction_name != "NOP"
    )

def is_load_type(instruction_name):
    return (
        instruction_name == "LB"
        or instruction_name == "LH"
        or instruction_name == "LW"
        or instruction_name == "LBU"
        or instruction_name == "LHU"
    )

def is_store_type(instruction_name):
    return (
        instruction_name == "SB"
        or instruction_name == "SH"
        or instruction_name == "SW"
    )

def is_mem_type(instruction_name):
    return (
        is_load_type(instruction_name)
        or is_store_type(instruction_name)
    )

def is_branch_type(instruction_name):
    return (
        instruction_name == "BEQ"
        or instruction_name == "BNE"
        or instruction_name == "BLT"
        or instruction_name == "BGE"
        or instruction_name == "BLTU"
        or instruction_name == "BGEU"
    )

def is_jump_type(instruction_name):
    return (
        instruction_name == "JAL"
        or instruction_name == "JALR"
    )

def is_upper_type(instruction_name):
    return (
        instruction_name == "LUI"
        or instruction_name == "AUIPC"
    )

def is_immediate_type(instruction_name):
    return (
        instruction_name == "JALR"
        or instruction_name == "ADDI"
        or instruction_name == "SLTI"
        or instruction_name == "SLTIU"
        or instruction_name == "XORI"
        or instruction_name == "ORI"
        or instruction_name == "ANDI"
    )

def is_immediate_shift_type(instruction_name):
    return (
        instruction_name == "SLLI"
        or instruction_name == "SRLI"
        or instruction_name == "SRAI"
    )

def is_register_type(instruction_name):
    return (
        instruction_name == "ADD"
        or instruction_name == "SUB"
        or instruction_name == "SLL"
        or instruction_name == "SLT"
        or instruction_name == "SLTU"
        or instruction_name == "XOR"
        or instruction_name == "SRL"
        or instruction_name == "SRA"
        or instruction_name == "OR"
        or instruction_name == "AND"
    )

def uses_rs1(instruction_name):
    return (
        not is_upper_type(instruction_name) 
        and instruction_name != "JAL"
        and instruction_name != "ECALL"
        and instruction_name != "NOP"
    )

def uses_rs2(instruction_name):
    return (
        is_branch_type(instruction_name) 
        or is_store_type(instruction_name) 
        or is_register_type(instruction_name)
    )

def get_print_instruction(instruction, instr_num = None):
    index_out = f'{instr_num}. ' if instr_num else ""
    print_out = None

    if(instruction.name == "N/A"): return None
    elif(instruction.name == "ECALL"):
        return f'{index_out}ECALL'

    rd = bin_to_dec(instruction.rd)
    rs1 = bin_to_dec(instruction.rs1)
    rs2 = bin_to_dec(instruction.rs2)
    imm = dec_to_hex(bin_to_dec(instruction.imm), 8)
    shamt = dec_to_hex(bin_to_dec(instruction.rd), 2)
    
    if(is_upper_type(instruction.name) or instruction.name == "JAL"):
        print_out = f'{index_out}{instruction.name} x{rd}, 0x{imm}'

    elif(is_branch_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rs1}, x{rs2}, 0x{imm}'

    elif(is_load_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rd}, 0x{imm}(x{rs1})'

    elif(is_store_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rs2}, 0x{imm}(x{rs1})'
    
    elif(is_immediate_shift_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rd}, x{rs1}, 0x{shamt}'

    elif(is_immediate_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rd}, x{rs1}, 0x{imm}'

    elif(is_register_type(instruction.name)):
        print_out = f'{index_out}{instruction.name} x{rd}, x{rs1}, x{rs2}'

    elif(instruction.name == "NOP"):
        print_out = f'{index_out}{instruction.name}'
    
    return print_out

def print_instruction(instruction, instr_num = None):
    print_out = f'{get_print_instruction(instruction, instr_num)} (PC: {dec_to_hex(bin_to_dec(instruction.pc), 8)})'
    if(not print_out): return
    else:
        print(print_out)
        return


'''Getters: Instruction Binary Extraction Functions'''
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

def immediate_generator(instruction):
    imm = ""

    # U-TYPE
    if (
        instruction.name == "LUI"
        or instruction.name == "AUIPC"
    ):
        imm = zero_extend(instruction.binary[-32:-12], 11)

    # J-TYPES
    elif (instruction.name == "JAL"):
        imm = sign_extend(zero_extend(instruction.binary[-32] + instruction.binary[-20:-12] + instruction.binary[-21] + instruction.binary[-31:-21], 0), 20)

    # I-TYPES
    elif (
        instruction.name == "JALR"
        or instruction.name == "LB"
        or instruction.name == "LH"
        or instruction.name == "LW"
        or instruction.name == "LBU"
        or instruction.name == "LHU"
        or instruction.name == "ADDI"
        or instruction.name == "SLTI"
        or instruction.name == "SLTIU"
        or instruction.name == "XORI"
        or instruction.name == "ORI"
        or instruction.name == "ANDI"
    ):
        imm = sign_extend(instruction.binary[-32:-20], 11)
    
    # B-TYPES   
    elif(
        instruction.name == "BEQ"
        or instruction.name == "BNE"
        or instruction.name == "BLT"
        or instruction.name == "BGE"
        or instruction.name == "BLTU"
        or instruction.name == "BGEU"
    ):
        temp_string = instruction.binary[-32] + instruction.binary[-8] + instruction.binary[-31:-25] + instruction.binary[-12:-8]
        imm = sign_extend(zero_extend(temp_string, 0), 12)

    # S-TYPES
    elif(
        instruction.name == "SB"
        or instruction.name == "SH"
        or instruction.name == "SW"
    ):
        temp_string = instruction.binary[-32:-25] + instruction.binary[-12:-7]
        imm = sign_extend(temp_string, 11)
        
    # R-TYPES
    elif(
        instruction.name == "ADD"
        or instruction.name == "SUB"
        or instruction.name == "SLL"
        or instruction.name == "SLT"
        or instruction.name == "SLTU"
        or instruction.name == "XOR"
        or instruction.name == "SRL"
        or instruction.name == "SRA"
        or instruction.name == "OR"
        or instruction.name == "AND"
        or instruction.name == "SLLI"
        or instruction.name == "SRLI"
        or instruction.name == "SRAI"
        or instruction.name == "NOP"
    ):
        imm = sign_extend("0", 0)

    return imm
    
def get_shamt(instruction_bin):
    return instruction_bin[-25:-20]


'''Utility Functions: Bitwise Manipulations/Operations and Radix Conversions'''
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



'''RISC-V PROCESSOR STAGES:'''
def fetch(pipeline, hex_file_path, pc, stall, first=False):
    [f_instruction, d_instruction, x_instruction, m_instruction, w_instruction] = pipeline.queue

    ''' Gets the instruction 1 clock cycle after is is done computing for the relevant fetch pc information'''
    # print("FETCH STALL CHECK:")
    # print (stall)
    if(not stall):
        if(is_branch_type(m_instruction.name) and m_instruction.pc_sel == "1"):
            # print("FETCH BRANCH (memory ALU)")
            pc = bin_to_dec(m_instruction.alu_result)
        elif(is_jump_type(x_instruction.name)):
            # print("FETCH JUMP (execute ALU)")
            pc = bin_to_dec(x_instruction.alu_result)
        elif(not first):
            # print("FETCH PC+4")
            pc += 4

    hex_file = glob.glob(hex_file_path)
    if(not hex_file):
        return False
    
    # print(int((pc - hex_to_dec(SP_BASE))/4))
    
    instruction_hex = "0" * 8
    for file in hex_file:
        with open(file, 'r') as f:
            instruction_hex = f.readlines()[int((pc - hex_to_dec(SP_BASE))/4)]
    # print(instruction_hex)
    # print(int((pc - hex_to_dec(SP_BASE))/4))

    f_instruction.pc = dec_to_bin(pc, 32)
    f_instruction.binary = dec_to_bin(hex_to_dec(instruction_hex), 32)

    return True


def fetch_check(pipeline, line, stall):
    '''Checks the state of the pipeline one clock cycle after, during current fetch'''
    # External: In trace
    args = get_trace_args(line)
    pc = dec_to_bin(hex_to_dec(args[0]), 32)
    instruction_bin = dec_to_bin(hex_to_dec(args[1]), 32)

    # Internal: In processor
    [f_instruction, d_instruction, x_instruction, m_instruction, w_instruction] = pipeline.queue

    res = True
    err = None

    operation = "PC+4"
    if(is_jump_type(x_instruction.name)):
        operation = "JUMP ALU"
    elif(is_branch_type(m_instruction.name) and m_instruction.pc_sel == "1"):
        operation = "BRANCH ALU"

    stalled = "(STALLED EXPECTED) " if stall else ""

    if(pc != f_instruction.pc): 
        res = False
        err = (f'<FETCH>: {stalled}pc: Got: {dec_to_hex(bin_to_dec(pc), 8)}, Expected: {dec_to_hex(bin_to_dec(f_instruction.pc), 8)} (Expected operation: {operation})')
    elif(instruction_bin != f_instruction.binary):
        res = False
        err = (f'<FETCH>: {stalled}instruction (binary): Got: {instruction_bin}, Expected: {f_instruction.binary}')

    return (res, err)

def decode(instruction, reg_file):
    instruction_name = "N/A"
    opcode = get_opcode(instruction.binary)
    rd = get_rd(instruction.binary)
    rs1 = get_rs1(instruction.binary)
    rs2 = get_rs2(instruction.binary)
    funct3 = get_funct3(instruction.binary)
    funct7 = get_funct7(instruction.binary)
    # imm = immediate_generator(instruction.binary)
    shamt = get_shamt(instruction.binary)
    rs1_data = reg_file.get(bin_to_dec(rs1))
    rs2_data = reg_file.get(bin_to_dec(rs2))

    # U-TYPES
    if(opcode == "0110111"): instruction_name = "LUI"
    elif(opcode == "0010111"): instruction_name = "AUIPC"
    
    # J-TYPE
    elif(opcode == "1101111"): instruction_name = "JAL"
    
    # B-TYPES
    elif(opcode == "1100011" and funct3 == "000"): instruction_name = "BEQ"
    elif(opcode == "1100011" and funct3 == "001"): instruction_name = "BNE"
    elif(opcode == "1100011" and funct3 == "100"): instruction_name = "BLT"
    elif(opcode == "1100011" and funct3 == "101"): instruction_name = "BGE"
    elif(opcode == "1100011" and funct3 == "110"): instruction_name = "BLTU"
    elif(opcode == "1100011" and funct3 == "111"): instruction_name = "BGEU"

    # I-TYPES
    elif(opcode == "1100111"): instruction_name = "JALR"
    elif(opcode == "0000011" and funct3 == "000"): instruction_name = "LB"
    elif(opcode == "0000011" and funct3 == "001"): instruction_name = "LH"
    elif(opcode == "0000011" and funct3 == "010"): instruction_name = "LW"
    elif(opcode == "0000011" and funct3 == "100"): instruction_name = "LBU"
    elif(opcode == "0000011" and funct3 == "101"): instruction_name = "LHU"
    elif(opcode == "0010011" and funct3 == "000"): instruction_name = "ADDI"
    elif(opcode == "0010011" and funct3 == "010"): instruction_name = "SLTI"
    elif(opcode == "0010011" and funct3 == "011"): instruction_name = "SLTIU"
    elif(opcode == "0010011" and funct3 == "100"): instruction_name = "XORI"
    elif(opcode == "0010011" and funct3 == "110"): instruction_name = "ORI"
    elif(opcode == "0010011" and funct3 == "111"): instruction_name = "ANDI"
    elif(opcode == "0010011" and funct3 == "001"): instruction_name = "SLLI"
    elif(opcode == "0010011" and funct3 == "101" and funct7 == "0000000"): instruction_name = "SRLI"
    elif(opcode == "0010011" and funct3 == "101" and funct7 == "0100000"): instruction_name = "SRAI"

    # S-TYPES
    elif(opcode == "0100011" and funct3 == "000"): instruction_name = "SB"
    elif(opcode == "0100011" and funct3 == "001"): instruction_name = "SH"
    elif(opcode == "0100011" and funct3 == "010"): instruction_name = "SW"

    # R-TYPES
    elif(opcode == "0110011" and funct3 == "000" and funct7 == "0000000"): instruction_name = "ADD"
    elif(opcode == "0110011" and funct3 == "000" and funct7 == "0100000"): instruction_name = "SUB"
    elif(opcode == "0110011" and funct3 == "001"): instruction_name = "SLL"
    elif(opcode == "0110011" and funct3 == "010"): instruction_name = "SLT"
    elif(opcode == "0110011" and funct3 == "011"): instruction_name = "SLTU"
    elif(opcode == "0110011" and funct3 == "100"): instruction_name = "XOR"
    elif(opcode == "0110011" and funct3 == "101" and funct7 == "0000000"): instruction_name = "SRL"
    elif(opcode == "0110011" and funct3 == "101" and funct7 == "0100000"): instruction_name = "SRA"
    elif(opcode == "0110011" and funct3 == "110"): instruction_name = "OR"
    elif(opcode == "0110011" and funct3 == "111"): instruction_name = "AND"
    
    elif(opcode == "1110011" and funct3 == "000" and funct7 == "0000000"): instruction_name = "ECALL"
    elif(opcode == "0000000"): instruction_name = "NOP"
    else: 
        instruction_name = "N/A"
        return False
    
    instruction.name = instruction_name
    instruction.opcode = opcode
    instruction.rd = rd
    instruction.rs1 = rs1
    instruction.rs2 = rs2
    instruction.funct3 = funct3
    instruction.funct7 = funct7
    instruction.shamt = shamt
    instruction.rs1_data = rs1_data
    instruction.rs2_data = rs2_data

    imm = immediate_generator(instruction)
    instruction.imm = imm

    if (instruction_name == "JALR"):
        instruction.alu_result = dec_to_bin(bin_to_dec(instruction.imm) + bin_to_dec(instruction.rs1_data), 32)
        # print(f'JALR to {instruction.alu_result}')
    elif (instruction_name == "JAL"):
        instruction.alu_result = dec_to_bin(bin_to_dec(instruction.imm) + bin_to_dec(instruction.pc), 32)
        # print(f'JAL to {instruction.alu_result}')

    # print("Instruction Immediate: ")
    # print (instruction.imm)
    # print("Instruction PC: ")
    # print (instruction.pc)
    # print("ALU Result: ")
    # print (instruction.alu_result)


    # print("Decode Instruction")
    # print(get_print_instruction(instruction))
    # print("Decode RS1")
    # print(instruction.rs1)
    # print("Decode RS2")
    # print(instruction.rs2)

    # print("Decode RS1 Data")
    # print(instruction.rs1_data)
    # print("Decode RS2 Data")
    # print(instruction.rs2_data)

    # print("Decode ID: ")
    # print(instruction.id)

    return True

def decode_check(instruction, line):
    args = get_trace_args(line)

    # External: In trace file
    pc = dec_to_bin(hex_to_dec(args[0]), 32)
    opcode = dec_to_bin(hex_to_dec(args[1]), 7)
    rd = dec_to_bin(hex_to_dec(args[2]), 5)
    rs1 = dec_to_bin(hex_to_dec(args[3]), 5)
    rs2 = dec_to_bin(hex_to_dec(args[4]), 5)
    funct3 = dec_to_bin(hex_to_dec(args[5]), 3)
    funct7 = dec_to_bin(hex_to_dec(args[6]), 7)
    imm = dec_to_bin(hex_to_dec(args[7]), 32)
    shamt = dec_to_bin(hex_to_dec(args[8]), 5)

    res = True
    err = None

    # PC CHECK!!!
    if(pc != instruction.pc):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - incorrect PC in DECODE stage. Got: {pc}. Expected {instruction.pc}'
        )

    # U-TYPES & J-TYPE
    if (
        (
            instruction.name == "LUI"
            or instruction.name == "AUIPC"
            or instruction.name == "JAL"
        )
        and not
        (
            opcode == instruction.opcode
            and rd == instruction.rd
            and imm == instruction.imm
        )
    ):        
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nrd: Got: {rd}, Expected: {instruction.rd}.'
            f'\nimm: Got: {imm}, Expected: {instruction.imm}.'
        )
        
    # I-TYPES
    elif (
        (
            instruction.name == "JALR"
            or instruction.name == "LB"
            or instruction.name == "LH"
            or instruction.name == "LW"
            or instruction.name == "LBU"
            or instruction.name == "LHU"
            or instruction.name == "ADDI"
            or instruction.name == "SLTI"
            or instruction.name == "SLTIU"
            or instruction.name == "XORI"
            or instruction.name == "ORI"
            or instruction.name == "ANDI"
        )
        and not
        (
            opcode == instruction.opcode
            and rd == instruction.rd
            and funct3 == instruction.funct3
            and rs1 == instruction.rs1
            and imm == instruction.imm
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nrd: Got: {rd}, Expected: {instruction.rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {instruction.funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {instruction.rs1}.'
            f'\nimm: Got: {imm}, Expected: {instruction.imm}.'
        )
    
    # B-TYPES   
    elif(
        (
            instruction.name == "BEQ"
            or instruction.name == "BNE"
            or instruction.name == "BLT"
            or instruction.name == "BGE"
            or instruction.name == "BLTU"
            or instruction.name == "BGEU"
        )
        and not
        (
            opcode == instruction.opcode
            and imm == instruction.imm
            and funct3 == instruction.funct3
            and rs1 == instruction.rs1
            and rs2 == instruction.rs2
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nimm: Got: {imm}, Expected: {instruction.imm}.'
            f'\nfunct3: Got: {funct3}, Expected: {instruction.funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {instruction.rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {instruction.rs2}.'
        )

    # S-TYPES
    elif(
        (
            instruction.name == "SB"
            or instruction.name == "SH"
            or instruction.name == "SW"
        )
        and not
        (
            opcode == instruction.opcode
            and imm == instruction.imm
            and rs1 == instruction.rs1
            and rs2 == instruction.rs2
            and funct3 == instruction.funct3
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nimm: Got: {imm}, Expected: {instruction.imm}.'
            f'\nfunct3: Got: {funct3}, Expected: {instruction.funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {instruction.rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {instruction.rs2}.'
        )
        
    # R-TYPES with SHAMT
    elif(
        (
            instruction.name == "SLLI"
            or instruction.name == "SRLI"
            or instruction.name == "SRAI"
        )
        and not
        (
            opcode == instruction.opcode
            and rd == instruction.rd
            and funct3 == instruction.funct3
            and rs1 == instruction.rs1
            and shamt == instruction.shamt
            and funct7 == instruction.funct7
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nrd: Got: {rd}, Expected: {instruction.rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {instruction.funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {instruction.rs1}.'
            f'\nshamt: Got: {shamt}, Expected: {instruction.shamt}.'
            f'\nfunct7: Got: {funct7}, Expected: {instruction.funct7}.'
        )

    # R-TYPES
    elif(
        (
            instruction.name == "ADD"
            or instruction.name == "SUB"
            or instruction.name == "SLL"
            or instruction.name == "SLT"
            or instruction.name == "SLTU"
            or instruction.name == "XOR"
            or instruction.name == "SRL"
            or instruction.name == "SRA"
            or instruction.name == "OR"
            or instruction.name == "AND"
        )
        and not
        (
            opcode == instruction.opcode
            and rd == instruction.rd
            and funct3 == instruction.funct3
            and rs1 == instruction.rs1
            and rs2 == instruction.rs2
            and funct7 == instruction.funct7
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nrd: Got: {rd}, Expected: {instruction.rd}.'
            f'\nfunct3: Got: {funct3}, Expected: {instruction.funct3}.'
            f'\nrs1: Got: {rs1}, Expected: {instruction.rs1}.'
            f'\nrs2: Got: {rs2}, Expected: {instruction.rs2}.'
            f'\nfunct7: Got: {funct7}, Expected: {instruction.funct7}.'
        )
    
    elif (
        instruction.name == "ECALL"
        and not 
        (
            opcode == instruction.opcode
            and instruction.binary[-32:-7] == "0" * 25
        )
    ):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - opcode: Got: {opcode}, Expected: {instruction.opcode}.'
            f'\nOther upper bits: Got: {instruction.binary[-32:-7]}, Expected: {"0" * 25}.'
        )
    
    return (res, err)


def execute(pipeline):
    [f_instruction, d_instruction, x_instruction, m_instruction, w_instruction] = pipeline.queue

    pc = x_instruction.pc
    rs1 = x_instruction.rs1_data
    rs2 = x_instruction.rs2_data
    imm = x_instruction.imm
    shamt = x_instruction.shamt
    
    alu_res = "0" * 32
    br_taken = "0"

    if(not (is_upper_type(x_instruction.name) or x_instruction.name == "JAL")):
        if(x_instruction.rs1 == m_instruction.rd and is_reg_writeback_type(m_instruction.name) and m_instruction.rd != "0" * 5):
            '''MX Bypass'''
            # print(f'MX BYPASS: rs1: [{get_print_instruction(m_instruction)} -> {get_print_instruction(x_instruction)}] (PC: {dec_to_hex(bin_to_dec(m_instruction.pc), 8)}) -> (PC: {dec_to_hex(bin_to_dec(x_instruction.pc), 8)}) (ID: {m_instruction.id}) -> (ID: {x_instruction.id}) (Bypass Value: {dec_to_hex(bin_to_dec(m_instruction.alu_result), 8)})')
            rs1 = m_instruction.alu_result
        elif(x_instruction.rs1 == w_instruction.rd and is_reg_writeback_type(w_instruction.name) and w_instruction.rd != "0" * 5):
            '''WX Bypass'''
            # print(f'WX BYPASS: rs1: [{get_print_instruction(w_instruction)} -> {get_print_instruction(x_instruction)}] (PC: {dec_to_hex(bin_to_dec(w_instruction.pc), 8)}) -> (PC: {dec_to_hex(bin_to_dec(x_instruction.pc), 8)}) (ID: {w_instruction.id}) -> (ID: {x_instruction.id}) (Bypass Value: {dec_to_hex(bin_to_dec(w_instruction.write_data), 8)})')
            rs1 = w_instruction.write_data
            

    if(is_register_type(x_instruction.name) or is_branch_type(x_instruction.name)):
        if(x_instruction.rs2 == m_instruction.rd and is_reg_writeback_type(m_instruction.name) and m_instruction.rd != "0" * 5):
            '''MX Bypass'''
            # print(f'MX BYPASS: rs2: [{get_print_instruction(m_instruction)} -> {get_print_instruction(x_instruction)}] (PC: {dec_to_hex(bin_to_dec(m_instruction.binary), 8)}) -> (PC: {dec_to_hex(bin_to_dec(x_instruction.binary), 8)}) (ID: {m_instruction.id}) -> (ID: {x_instruction.id}) (Bypass Value: {dec_to_hex(bin_to_dec(m_instruction.alu_result), 8)})')
            rs2 = m_instruction.alu_result
        elif(x_instruction.rs2 == w_instruction.rd and is_reg_writeback_type(w_instruction.name) and w_instruction.rd != "0" * 5):
            '''WX Bypass'''
            # print(f'WX BYPASS: rs2: [{get_print_instruction(w_instruction)} -> {get_print_instruction(x_instruction)}] (PC: {dec_to_hex(bin_to_dec(w_instruction.binary), 8)}) -> (PC: {dec_to_hex(bin_to_dec(x_instruction.binary), 8)}) (ID: {w_instruction.id}) -> (ID: {x_instruction.id}) (Bypass Value: {dec_to_hex(bin_to_dec(w_instruction.write_data), 8)})')
            rs2 = w_instruction.write_data
    
    # print(f'rs1: {rs1}')

    # print("rs1: ")
    # print(rs1)

    # print("rs2: ")
    # print(rs2)

    # print("imm: ")
    # print(imm)



    # print("Fetch ID: ")
    # print(f_instruction.id)
    # # print("Decode ID: ")
    # # print(d_instruction.id)
    # # print("Execute ID: ")
    # # print(x_instruction.id)
    # print("Memory ID: ")
    # print(m_instruction.id)
    # print("Writeback ID: ")
    # print(w_instruction.id)

    # J-Types - take the branch (pc = ALU output)
    if(x_instruction.name == "JAL" or x_instruction.name == "JALR"):
        br_taken = "1"


    # U-TYPES
    if (x_instruction.name == "LUI"): alu_res = zero_extend(imm[-32:-12], 11)
    
    # Add IMM to PC
    elif (
        x_instruction.name == "AUIPC"
        # or x_instruction.name == "JAL"
    ): 
        alu_res = dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32)
        
    # B-TYPES
    elif(
        x_instruction.name == "BEQ"
        or x_instruction.name == "BNE"
        or x_instruction.name == "BLT"
        or x_instruction.name == "BGE"
        or x_instruction.name == "BLTU"
        or x_instruction.name == "BGEU"
    ):
        if (x_instruction.name == "BEQ"): br_taken = "1" if rs1 == rs2 else "0"
        elif (x_instruction.name == "BNE"): br_taken = "1" if not(rs1 == rs2) else "0"
        elif (x_instruction.name == "BLT"): br_taken = "1" if signed_less_than(rs1, rs2) else "0"
        elif (x_instruction.name == "BGE"): br_taken = "1" if not(signed_less_than(rs1, rs2)) else "0"
        elif (x_instruction.name == "BLTU"): br_taken = "1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0"
        elif (x_instruction.name == "BGEU"): br_taken = "1" if not(bin_to_dec(rs1) < bin_to_dec(rs2)) else "0"

        # Always calculate ALU result for branch
        alu_res = dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32)

    # I-TYPES
    elif (
        # x_instruction.name == "JALR"
        x_instruction.name == "LB"
        or x_instruction.name == "LW"
        or x_instruction.name == "LH"
        or x_instruction.name == "LBU"
        or x_instruction.name == "LHU"
        or x_instruction.name == "SB"
        or x_instruction.name == "SH"
        or x_instruction.name == "SW"
        or x_instruction.name == "ADDI"
    ):
        # print("EXECUTE CHECK:")
        # print(rs1)
        # print(imm)
        # print(dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32))
        # print()
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32)

    elif (x_instruction.name == "SLTI"):
        alu_res = "0" * 31 + ("1" if signed_less_than(rs1, imm) else "0")

    elif (x_instruction.name == "SLTIU"):
        alu_res = "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(imm) else "0")

    elif (x_instruction.name == "XORI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(imm), 32)

    elif (x_instruction.name == "ORI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) | bin_to_dec(imm), 32)

    elif (x_instruction.name == "ANDI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) & bin_to_dec(imm), 32)

    elif (x_instruction.name == "SLLI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) << bin_to_dec(shamt), 32)

    elif (x_instruction.name == "SRLI"):
        alu_res = dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 32)

    elif (x_instruction.name == "SRAI"):
        alu_res = rs1[0] * bin_to_dec(shamt) + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 32 - bin_to_dec(shamt))


    # R-TYPES
    elif (x_instruction.name == "ADD"):
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(rs2), 32)

    elif (x_instruction.name == "SUB"):
        # Two's complement method of adding
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(twos_complement(rs2)), 32)

    elif (x_instruction.name == "SLL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) << bin_to_dec(rs2[-5:]), 32)

    elif (x_instruction.name == "SLT"):
        alu_res = "0" * 31 + ("1" if signed_less_than(rs1, rs2) else "0")

    elif (x_instruction.name == "SLTU"):
        alu_res = "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0")

    elif (x_instruction.name == "XOR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(rs2), 32)

    elif (x_instruction.name == "SRL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2[-5:]), 32)

    elif (x_instruction.name == "SRA"):
        alu_res = rs1[0] * bin_to_dec(rs2[-5:]) + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2[-5:]), 32 - bin_to_dec(rs2[-5:]))

    elif (x_instruction.name == "OR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) | bin_to_dec(rs2), 32)

    elif (x_instruction.name == "AND"):
        alu_res = dec_to_bin(bin_to_dec(rs1) & bin_to_dec(rs2), 32)

    # elif (instruction == "ECALL"):
    x_instruction.alu_result = alu_res
    x_instruction.pc_sel = br_taken
    return True

def execute_check(instruction, line):
    res = True
    err = None
        
    # External: In trace file
    exec_args = get_trace_args(line)
    pc = dec_to_bin(hex_to_dec(exec_args[0]), 32)
    alu_res = dec_to_bin(hex_to_dec(exec_args[1]), 32)
    br_taken = dec_to_bin(hex_to_dec(exec_args[2]), 1)

    # Internal: In processor
    exp_alu_res = instruction.alu_result
    exp_br_taken = instruction.pc_sel

    # PC CHECK!!!
    if(pc != instruction.pc):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - incorrect PC in EXECUTE stage. Got: {pc}. Expected {instruction.pc}'
        )
    
    # B-TYPES
    if (
        (
            instruction.name == "BEQ"
            or instruction.name == "BNE"
            or instruction.name == "BLT"
            or instruction.name == "BGE"
            or instruction.name == "BLTU"
            or instruction.name == "BGEU"
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
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - branch_taken: {br_taken}, expected: {exp_br_taken}.'
            + f'\n{instruction.name}: alu_result: got: {alu_res}, expected (if branch taken): {exp_alu_res}'
        )
    
    # Other Types
    # ???: branch_taken is a DC for jumps/J-Types?
    elif (
        (
            instruction.name == "LUI"
            or instruction.name == "AUIPC"
            or instruction.name == "JAL" 
            or instruction.name == "JALR"
            or instruction.name == "LB"
            or instruction.name == "LW"
            or instruction.name == "LH"
            or instruction.name == "LBU"
            or instruction.name == "LHU"
            or instruction.name == "SB"
            or instruction.name == "SH"
            or instruction.name == "SW"
            or instruction.name == "ADDI"
            or instruction.name == "SLTI"
            or instruction.name == "SLTIU"
            or instruction.name == "XORI"
            or instruction.name == "ORI"
            or instruction.name == "ANDI"
            or instruction.name == "SLLI"
            or instruction.name == "SRLI"
            or instruction.name == "SRAI"
            or instruction.name == "ADD"
            or instruction.name == "SUB"
            or instruction.name == "SLL"
            or instruction.name == "SLT"
            or instruction.name == "SLTU"
            or instruction.name == "XOR"
            or instruction.name == "SRL"
            or instruction.name == "SRA"
            or instruction.name == "OR"
            or instruction.name == "AND"
        )
        and alu_res != exp_alu_res
    ): 
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - alu_result: got: {alu_res}, expected: {exp_alu_res}'

    elif (instruction.name == "ECALL"):
        # ECALL Warning!
        res = True
        if(instruction.binary[-32:-7] != "0" * (25)):
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - WARNING - upper 25 bits are not all 0s: got: {instruction.binary[-32:-7]}, expected: {"0" * (25)}'

    return (res, err)


def register_check(instruction, line):
    res = True
    err = None

    # External: In trace
    args = get_trace_args(line)
    addr_rs1 = dec_to_bin(hex_to_dec(args[0]), 5)
    addr_rs2 = dec_to_bin(hex_to_dec(args[1]), 5)
    rs1 = dec_to_bin(hex_to_dec(args[2]), 32) # in hex
    rs2 = dec_to_bin(hex_to_dec(args[3]), 32) # in hex

    # Internal: In processor
    exp_addr_rs1 = instruction.rs1
    exp_addr_rs2 = instruction.rs2
    exp_rs1 = instruction.rs1_data
    exp_rs2 = instruction.rs2_data
    
    # Verify addresses
    # For instructions utilizing rs1 and/or rs2
    if(
        not (
            is_upper_type(instruction.name)
            or instruction.name == "JAL"
        )
        and
        addr_rs1 != exp_addr_rs1
    ):
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - Incorrect rs1 address. Got: {addr_rs1}. Expected: {exp_addr_rs1}'
    elif(
        not (
            is_upper_type(instruction.name)
            or instruction.name == "JAL"
            or is_load_type(instruction.name)
            or is_immediate_type(instruction.name)
            or is_immediate_shift_type(instruction.name)
        )
        and
        addr_rs2 != exp_addr_rs2
    ):
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - Incorrect rs2 address. Got: {addr_rs2}. Expected: {exp_addr_rs2}'

    # Verify register values
    elif(
        not (
            is_upper_type(instruction.name)
            or instruction.name == "JAL"
        )
        and
        rs1 != exp_rs1
    ):
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - Incorrect rs1 data at address {addr_rs1}. Got: {rs1}. Expected: {exp_rs1}.'
    elif(
        not (
            is_upper_type(instruction.name)
            or instruction.name == "JAL"
            or is_load_type(instruction.name)
            or is_immediate_type(instruction.name)
            or is_immediate_shift_type(instruction.name)
        )
        and
        rs2 != exp_rs2
    ):
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - Incorrect rs2 data at address {addr_rs2}. Got: {rs2}. Expected: {exp_rs2}.'

    return (res, err)


# Executes stores, loads can be access later during memory checking
def memory(pipeline, dmemory):
    [f_instruction, d_instruction, x_instruction, m_instruction, w_instruction] = pipeline.queue
    
    out = None

    mem_index = bin_to_dec(m_instruction.alu_result[-32:]) - hex_to_dec(SP_BASE)
    write_data = m_instruction.rs2_data
    memory_access_size = m_instruction.funct3[-2:]
    mem_rw = "0"
    reg_w_en = "0"

    #// TODO: compute and store memory_access_size into instruction
    #// TODO: compute and store mem_rw into instruction

    if(
        m_instruction.rs2 == w_instruction.rd
        and w_instruction.rd != "0" * 5
        and is_store_type(m_instruction.name) 
        and is_reg_writeback_type(w_instruction.name)
    ):
        '''WM Bypass'''
        # print(f'WM BYPASS: rs2: [{get_print_instruction(w_instruction)} -> {get_print_instruction(m_instruction)}] (PC: {dec_to_hex(bin_to_dec(w_instruction.pc), 8)}) -> (PC: {dec_to_hex(bin_to_dec(m_instruction.pc), 8)}) (ID: {w_instruction.id}) -> (ID: {m_instruction.id})')
        write_data = w_instruction.write_data

    if(m_instruction.name == "SB"):
        # out = write_data[-8:]
        out = write_data
        dmemory.set_byte(write_data, mem_index)
    elif(m_instruction.name == "SH"):
        # out = write_data[-16:]
        out = write_data
        dmemory.set_half(write_data, mem_index)
    elif(m_instruction.name == "SW"):
        # out = write_data[-32:]
        out = write_data
        dmemory.set_word(write_data, mem_index)
    elif(m_instruction.name == "LB"): 
        out = sign_extend(dmemory.get_byte(mem_index), 7)
    elif(m_instruction.name == "LH"): 
        out = sign_extend(dmemory.get_half(mem_index), 15)
    elif(m_instruction.name == "LW"): 
        out = dmemory.get_word(mem_index)[-32:]
    elif(m_instruction.name == "LBU"): 
        out = "0" * 24 + dmemory.get_byte(mem_index)[-8:]
    elif(m_instruction.name == "LHU"): 
        out = "0" * 16 + dmemory.get_half(mem_index)[-16:]

    if(is_store_type(m_instruction.name)):
        mem_rw = "1"
    
    if(is_load_type(m_instruction.name)):
        reg_w_en = "1"

    '''PC+4 and ALU are accessible in Memory stage'''
    calc = None
    # WB: PC+4
    if(
        m_instruction.name == "JAL"
        or m_instruction.name == "JALR"
    ):
        calc = dec_to_bin(bin_to_dec(m_instruction.pc) + 4, 32)
        reg_w_en = "1"
    # WB: ALU
    elif (
        m_instruction.name == "LUI"
        or m_instruction.name == "AUIPC"
        or m_instruction.name == "ADDI"
        or m_instruction.name == "SLTI"
        or m_instruction.name == "SLTIU"
        or m_instruction.name == "XORI"
        or m_instruction.name == "ORI"
        or m_instruction.name == "ANDI"
        or m_instruction.name == "SLLI"
        or m_instruction.name == "SRLI"
        or m_instruction.name == "SRAI"
        or m_instruction.name == "ADD"
        or m_instruction.name == "SUB"
        or m_instruction.name == "SLL"
        or m_instruction.name == "SLT"
        or m_instruction.name == "SLTU"
        or m_instruction.name == "XOR"
        or m_instruction.name == "SRL"
        or m_instruction.name == "SRA"
        or m_instruction.name == "OR"
        or m_instruction.name == "AND"
    ):
    # else:
       calc = m_instruction.alu_result
       reg_w_en = "1"
    
    elif(out):
        calc = out


    if(calc):
        m_instruction.write_data = calc

    # print(f'calc: {calc}')

    m_instruction.memory_access_size = memory_access_size
    m_instruction.memory_data = out
    m_instruction.mem_rw = mem_rw
    m_instruction.reg_w_en = reg_w_en

    return True

def memory_check(instruction, line, dmemory):
    res = True
    err = None
    
    # External: In trace file
    args = get_trace_args(line)
    pc = dec_to_bin(hex_to_dec(args[0]), 32)
    mem_addr = dec_to_bin(hex_to_dec(args[1]), 32)
    rw = args[2] == "1"
    access_size = args[3]
    data = dec_to_bin(hex_to_dec(args[4]), 32)

    # Internal: In processor
    exp_access_size = dec_to_hex(bin_to_dec(instruction.memory_access_size), 1)
    write_data = instruction.write_data

    # PC CHECK!!!
    if(pc != instruction.pc):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - incorrect PC in MEMORY stage. Got: {pc}. Expected {instruction.pc}'
        )

    # Verify access_size, address, read_write, and memory_data!
    if (is_mem_type(instruction.name)):
        '''access_size & address'''
        if(
            (
                instruction.name == "LB" 
                or instruction.name == "LBU"
                or instruction.name == "SB"
            )
            and access_size != exp_access_size
        ):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - access_size is {access_size}, expecting 0'

        elif(
            (
                instruction.name == "LH" 
                or instruction.name == "LHU"
                or instruction.name == "SH"
            )
            and access_size != exp_access_size
        ):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - access_size is {access_size}, expecting 1'

        elif(
            (
                instruction.name == "LW" 
                or instruction.name == "SW"
            )
            and access_size != exp_access_size
        ):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - access_size is {access_size}, expecting 2'
        

        elif(mem_addr != instruction.alu_result):
            '''address'''
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - trace address is not the same as memory address. addr: Got: {mem_addr}, Expected: {instruction.alu_result}'

    # Default - Read: rw = 0. On stores - Write: rw = 1
    # If it is not a store instruction, it should not be writing
    if(res):
        '''read_write'''
        if(
            rw
            and
            not (
                instruction.name == "SB"
                or instruction.name == "SH"
                or instruction.name == "SW"
            )
        ):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - should NOT be writing to dmemory if not a Store instruction (SB, SH, SW only)'

    if(res and is_mem_type(instruction.name)):
        '''memory_data'''
        addr_dec = bin_to_dec(instruction.alu_result[-32:]) - hex_to_dec(SP_BASE)
        dmem_data = dmemory.get_word(addr_dec)
        # Check if state of dmemory is correct on reads
        # Only If it is a memory instruction. DC otherwise
        if(is_load_type(instruction.name)):
            if(
                (
                    instruction.name == "LB" 
                    or instruction.name == "LBU"
                )  
                and data[-8:] != dmem_data[-8:]
            ):
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory byte at address {dec_to_hex(addr_dec, 8)}: got {data[-8:]}, expected: {dmem_data[-8:]}'

            elif(
                (
                    instruction.name == "LH"
                    or instruction.name == "LHU"
                )  
                and data[-16:] != dmem_data[-16:]
            ):
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory half-word at address {dec_to_hex(addr_dec, 8)}: got {data[-16:]}, expected: {dmem_data[-16:]}'


            elif(
                (
                    instruction.name == "LW"
                )
                and data[-32:] != dmem_data[-32:]
            ): 
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory word at address {dec_to_hex(addr_dec, 8)}: got {data[-32:]}, expected: {dmem_data[-32:]}'
        
        # Verify that data was stored correctly
        elif(is_store_type(instruction.name)):
            if(
                instruction.name == "SB"  
                and dmem_data[-8:] != write_data[-8:]
            ):
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory byte at address {dec_to_hex(addr_dec, 8)}: got: {dmem_data[-8:]}, expecting: {write_data[-8:]}'

            elif(
                instruction.name == "SH"  
                and dmem_data[-16:] != write_data[-16:]
            ):
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory half-word at address {dec_to_hex(addr_dec, 8)}: got: {dmem_data[-16:]}, expecting: {write_data[-16:]}'


            elif(
                instruction.name == "SW"
                and dmem_data[-32:] != write_data[-32:]
            ): 
                res = False
                err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - dmemory word at address {dec_to_hex(addr_dec, 8)}: got: {dmem_data[-32:]}, expecting: {write_data[-32:]}'

    return (res, err)


def write(instruction, reg_file):
    res = False

    if(instruction.reg_w_en == "1"): res = reg_file.set(instruction.write_data, bin_to_dec(instruction.rd))
    else: res = True
    
    return res
    
def write_check(instruction, line):
    res = True
    err = None

    # External: In Trace
    args = get_trace_args(line)
    pc = dec_to_bin(hex_to_dec(args[0]), 32)
    write_enable = args[1] == "1"
    write_rd = hex_to_dec(args[2])
    data_rd = dec_to_bin(hex_to_dec(args[3]), 32)

    # Internal: In RISC-V Processor
    rd_addr = bin_to_dec(instruction.rd)
    exp_write = instruction.write_data
    
    # print(f'{instruction.print()}')

    # PC CHECK!!!
    if(pc != instruction.pc):
        res = False
        err = (
            f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - incorrect PC in WRITE stage. Got: {pc}. Expected {instruction.pc}'
        )

    # Verify write_enable
    if(
        write_enable
        and (
            is_branch_type(instruction.name)
            or is_store_type(instruction.name)
            or instruction.name == "ECALL"
            or instruction.name == "NOP"
        )
    ):
        res = False
        err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - should NOT be writing to destination register, rd; (register) write_enable is "1", expecting "0"'
    elif(
        not (
            is_branch_type(instruction.name)
            or is_store_type(instruction.name)
            or instruction.name == "ECALL"
            or instruction.name == "NOP"
        )
        and rd_addr != 0
    ):
        if(not write_enable):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - should be writing to destination register, rd; (register) write_enable is "0", expecting "1"'

        elif(write_rd != rd_addr):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - WRITE - incorrect destination register address, rd; got {write_rd}, expecting {rd_addr}'

        elif(data_rd != exp_write):
            res = False
            err = f'<{instruction.name}>: {dec_to_hex(bin_to_dec(instruction.binary), 8)} - WRITE - incorrect write back data; got {data_rd}, expecting {exp_write}'
    
    return (res, err)



def main():
    parser.add_argument("trace", type=trace_file, help="Path to the a .trace file (ex. ../sim/verilator/test_pd/rv32ui-p-sltiu.trace)")
    parser.add_argument("hex", type=hex_file, help="Path to the a .x file (ex. /rv32-benchmarks/individual-instructions/rv32ui-p-sltiu.x). Used to initialize dmemory")
    parser.add_argument("-i", "--instructions", dest="instructions", action="store_true", help="Print all instructions in RISC-V format (Default: False, instructions enumerated)")
    parser.add_argument("-in", "--instructions-no-num", dest="instructions_no_num", action="store_true", help="Print all instructions in RISC-V format (Default: False, instructions NOT enumerated)")
    parser.add_argument("-r", "--regfile", dest="regfile", action="store_true", help="Print the state of the Register File at each [R] (Default: False)")
    # parser.add_argument("-t", "--terminate", "--ecall", dest="ecall", action="store_true", help="End (terminate) the parse.py check at the first ECALL instruction (Default: False)")
    parser.add_argument("-s", "--skip", dest="skip", action="store_false", help="CONTINUE execution on errors (Default: True)")
    parser.add_argument("-m", "--mem", dest="mem", type=int, default=1048576, metavar="MEM_DEPTH", help="Number of bytes (8-bit values) in data memory (dmemory). (Default: 1048576)")
    args = parser.parse_args()

    TRACE = args.trace
    HEX_FILE = args.hex
    SHOW_INSTR = args.instructions
    SHOW_INSTR_NO_NUM = args.instructions_no_num
    SHOW_REG = args.regfile
    # END_ON_ECALL = args.ecall
    END_ON_ECALL = True
    STOP_ON_ERR = args.skip
    MEM_DEPTH = args.mem # # of bytes (8-bit values) in decimal

    if MEM_DEPTH <= 0:
        print(f'Please enter a MEM_DEPTH greater than 0. Got: {MEM_DEPTH}\n')
        return
    
    pipeline = Pipeline()
    reg_file = RegFile(32, 32)
    dmemory = DMemory(MEM_DEPTH, 8, HEX_FILE) # Byte-addressable memory

    # Set stack pointer (sp = x2) to 0x01000000 + MEM_DEPTH    
    reg_file.set(dec_to_bin(hex_to_dec(SP_BASE) + MEM_DEPTH, 32), 2) 
    
    iterator = 0
    instr_num = 0

    f = open(TRACE, 'r')
    line = f.readline()
    
    # blank_fetch = Instruction("Fetch")
    # blank_decode = Instruction("Decode")
    # blank_execute = Instruction("Execute")
    # blank_memory = Instruction("Memory")
    # blank_write = Instruction("Write")
    # pipeline.add(blank_fetch)
    # pipeline.add(blank_decode)
    # pipeline.add(blank_execute)
    # pipeline.add(blank_memory)
    # pipeline.add(blank_write)
    blank_instruction = Instruction(instr_num)
    pipeline.add(blank_instruction)
    instr_num += 1

    stall = False
    prev_stall_id = None
    pc_index = hex_to_dec(SP_BASE)

    # TODO: Add line of Trace file where the error occurred!!!

    while (line):
        iterator += 1
        stage = line[1]
        [f_instruction, d_instruction, x_instruction, m_instruction, w_instruction] = pipeline.queue

        # print("===== START of parse loop =====")
        # pipeline.print()
        
        #// TODO: Update PC CHECK for each type of instruction (at the beginning)
        # i.e. if fetch: pc_check() = if(fetch_pc == pc) 
        # if(stall): print("STALLED")

        if(stage == 'F'):
            '''FETCH'''
            res = fetch(pipeline, HEX_FILE, pc_index, stall, iterator == 1)
            if(not res):
                print(f'The given hex file does not exist...')
            
            pc_index = bin_to_dec(f_instruction.pc)

            (success, err) = fetch_check(pipeline, line, stall)
            # stall = False

            if (not success):
                print(f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<FETCH> PC: {dec_to_hex(bin_to_dec(f_instruction.pc), 8)}, did not FETCH correctly.')
            if(err):
                print(err)
            if(STOP_ON_ERR and not success):
                print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                break

            # TODO: ADD FETCH CHECK to make sure that when there is a jump or branch, it's going to the right place!!!
            # By comparing the pc to that and the corresponding one in the .x file

            # Figure out how to get a specific line the .x file based on if jump/branch need to be taken
            # and extract the PC and compare it to the currently fetched PC
                
        # elif(stage == 'D' and not stall):
        elif(stage == 'D'):
            '''DECODE'''            
            # print("===== BEFORE decoding =====")
            # pipeline.print()

            res = decode(d_instruction, reg_file)
            
            # print("===== AFTER decoding =====")
            # pipeline.print()

            # print(stall)

            # If stall, print stall
            # Else print normally?

            # print(is_reg_writeback_type(w_instruction.name)
            #     and
            #     (
            #         # RS1
            #         w_instruction.rd == d_instruction.rs1
            #         and d_instruction.rs1 != "0" * 5
            #         and not (
            #             (
            #                 m_instruction.rd == d_instruction.rs1 
            #                 and is_reg_writeback_type(m_instruction.name)
            #             )
            #             or (
            #                 x_instruction.rd == d_instruction.rs1
            #                 and is_reg_writeback_type(x_instruction.name)
            #             )
            #         )
            #     )
            # )
            # print(is_reg_writeback_type(w_instruction.name))
            # print(w_instruction.rd == d_instruction.rs1
            #     and d_instruction.rs1 != "0" * 5
            #     and not (
            #         (
            #             m_instruction.rd == d_instruction.rs1 
            #             and is_reg_writeback_type(m_instruction.name)
            #         )
            #         or (
            #             x_instruction.rd == d_instruction.rs1
            #             and is_reg_writeback_type(x_instruction.name)
            #         )
            #     )
            # )

            # print(is_reg_writeback_type(w_instruction.name)
            #     and
            #     (
            #         # RS2
            #         w_instruction.rd == d_instruction.rs2
            #         and d_instruction.rs2 != "0" * 5
            #         and not (
            #             (
            #                 m_instruction.rd == d_instruction.rs2
            #                 and is_reg_writeback_type(m_instruction.name)
            #             )
            #             or (
            #                 x_instruction.rd == d_instruction.rs2
            #                 and is_reg_writeback_type(x_instruction.name)
            #             )
            #         )
            #     )
            # )
            # print(is_load_type(x_instruction)
            #     and (
            #         d_instruction.rs1 == x_instruction.rd
            #         or (d_instruction.rs2 == x_instruction.rd and not is_store_type(d_instruction.name))
            #     )
            # )
            # print(is_load_type(x_instruction.name))
            # print(d_instruction.rs1 == x_instruction.rd
            #     or (d_instruction.rs2 == x_instruction.rd and not is_store_type(d_instruction.name))
            # )

            # pipeline.print()

            #// TODO: stalling boolean logic
            stall = (
                # Decode-Writeback RAW
                (
                    is_reg_writeback_type(w_instruction.name)
                    and w_instruction.rd != "0" * 5
                    and
                    (
                        (
                            # RS1
                            w_instruction.rd == d_instruction.rs1
                            and d_instruction.rs1 != "0" * 5
                            and uses_rs1(d_instruction.name)
                            and not (
                                (
                                    m_instruction.rd == d_instruction.rs1 
                                    and is_reg_writeback_type(m_instruction.name)
                                )
                                or (
                                    x_instruction.rd == d_instruction.rs1
                                    and is_reg_writeback_type(x_instruction.name)
                                )
                            )
                        )
                        or
                        (
                            # RS2
                            w_instruction.rd == d_instruction.rs2
                            and d_instruction.rs2 != "0" * 5
                            and uses_rs2(d_instruction.name)
                            and not (
                                (
                                    m_instruction.rd == d_instruction.rs2
                                    and is_reg_writeback_type(m_instruction.name)
                                )
                                or (
                                    x_instruction.rd == d_instruction.rs2
                                    and is_reg_writeback_type(x_instruction.name)
                                )
                            )
                        )
                    )
                )
                or 
                # Load-To-Use Dependency (i.e. Decode: Depends on LOAD && Execute: LOAD)
                (
                    is_load_type(x_instruction.name)
                    and x_instruction.rd != "0" * 5
                    and not is_upper_type(d_instruction.name) 
                    and d_instruction.name != "JAL"
                    and d_instruction.name != "ECALL"
                    and d_instruction.name != "NOP"
                    and (
                        d_instruction.rs1 == x_instruction.rd and uses_rs1(d_instruction.name)
                        or (d_instruction.rs2 == x_instruction.rd and not is_store_type(d_instruction.name) and uses_rs2(d_instruction.name))
                    )
                )
                or
                # JALR: rs1 RAW: DX, DM, DW
                (
                    d_instruction.name == "JALR"
                    and (
                        (d_instruction.rs1 == x_instruction.rd and is_reg_writeback_type(x_instruction.name))
                        or (d_instruction.rs1 == m_instruction.rd and is_reg_writeback_type(m_instruction.name))
                        or (d_instruction.rs1 == w_instruction.rd and is_reg_writeback_type(w_instruction.name))
                    )
                )
                or
                # STORE data (SW, SH, SB): rs2 RAW: DM, DW
                # (DX is covered by WM bypass)
                (
                    is_store_type(d_instruction.name)
                    and (
                        (d_instruction.rs2 == m_instruction.rd and is_reg_writeback_type(m_instruction.name))
                        or (d_instruction.rs2 == w_instruction.rd and is_reg_writeback_type(w_instruction.name))
                    )
                )
            )


            # print(get_print_instruction(d_instruction))
            # print(get_print_instruction(x_instruction))
            # print(get_print_instruction(m_instruction))
            # print(get_print_instruction(w_instruction))
            # print(stall)
            # print((
            #         d_instruction.name == "JALR"
            #         and (
            #             (d_instruction.rs1 == x_instruction.rd and is_reg_writeback_type(x_instruction.name))
            #             or (d_instruction.rs1 == m_instruction.rd and is_reg_writeback_type(m_instruction.name))
            #             or (d_instruction.rs1 == w_instruction.rd and is_reg_writeback_type(w_instruction.name))
            #         )
            #     ))
            # print(
            #     (
            #         is_store_type(d_instruction.name)
            #         and (
            #             (d_instruction.rs2 == m_instruction.rd and is_reg_writeback_type(m_instruction.name))
            #             or (d_instruction.rs2 == w_instruction.rd and is_reg_writeback_type(w_instruction.name))
            #         )
            #     )
            # )
            # print((
            #         is_load_type(x_instruction.name)
            #         and x_instruction.rd != "0" * 5
            #         and is_reg_writeback_type(x_instruction.name)
            #         and (
            #             d_instruction.rs1 == x_instruction.rd
            #             or (d_instruction.rs2 == x_instruction.rd and not is_store_type(d_instruction.name))
            #         )
            #     ))
            # print(
            #     d_instruction.rs1 == x_instruction.rd
            # )
            # print(get_print_instruction(d_instruction))
            # print(d_instruction.rs1)
            # print(get_print_instruction(x_instruction))
            # print(x_instruction.rd)
            # print()


            # print(w_instruction.rd == d_instruction.rs2)
            # print(is_branch_type(d_instruction.name) 
            #         or is_store_type(d_instruction.name) 
            #         or is_register_type(d_instruction.name)
            # )
            # print(not (
            #         (
            #             m_instruction.rd == d_instruction.rs2
            #             and is_reg_writeback_type(m_instruction.name)
            #         )
            #         or (
            #             x_instruction.rd == d_instruction.rs2
            #             and is_reg_writeback_type(x_instruction.name)
            #         )
            #     ))
            # print( w_instruction.rd == d_instruction.rs2
            #     and d_instruction.rs2 != "0" * 5
            #     and (
            #         is_branch_type(d_instruction.name) 
            #         or is_store_type(d_instruction.name) 
            #         or is_register_type(d_instruction.name)
            #     )
            #     and not (
            #         (
            #             m_instruction.rd == d_instruction.rs2
            #             and is_reg_writeback_type(m_instruction.name)
            #         )
            #         or (
            #             x_instruction.rd == d_instruction.rs2
            #             and is_reg_writeback_type(x_instruction.name)
            #         )
            #     )
            # )
            # print(stall)
            # print(get_print_instruction(d_instruction))
            # print(get_print_instruction(w_instruction))



            # print(stall)
            # pipeline.print()



            # print("===== STALL CHECK =====")
            # pipeline.print()
            # print(is_load_type(x_instruction))
            # print( (w_instruction.rd == d_instruction.rs1 and d_instruction.rs1 != "0" * 5)
            #             or (w_instruction.rd == d_instruction.rs2 and d_instruction.rs2 != "0" * 5))

            # print()

            if(d_instruction.name != "NOP" and d_instruction.name != "N/A" 
            and ((prev_stall_id == d_instruction.id and stall) or prev_stall_id != d_instruction.id)):
                instr_num += 1

            if(instr_num > 1 and (SHOW_INSTR or SHOW_INSTR_NO_NUM)): 
                if(prev_stall_id != d_instruction.id):
                    '''During a back-to-back stall do not reprint the instruction stuck in the Decode stage.'''
                    print_instruction(d_instruction, instr_num - 1 if SHOW_INSTR else None)
                elif(stall):
                    if(SHOW_INSTR):
                        print(f'{instr_num}. STALL')
                    elif(SHOW_INSTR_NO_NUM):
                        print('STALL')
                
            
            if(stall): prev_stall_id = d_instruction.id
            else: prev_stall_id = None

            # pipeline.print()
            
            
            # print()

            success = True
            err = None
            if(d_instruction.name != "NOP"):
                (success, err) = decode_check(d_instruction, line)

            if(not success and d_instruction.name != "N/A"):
                print(
                    f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<DECODE> PC: {dec_to_hex(bin_to_dec(d_instruction.pc), 8)}. Instruction: {dec_to_hex(bin_to_dec(d_instruction.binary), 8)} - {get_print_instruction(d_instruction)}, did not DECODE correctly'
                )
            if(err):
                print(err)
            if(STOP_ON_ERR and not success and d_instruction.name != "N/A"):
                print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                break
            if(END_ON_ECALL and d_instruction.name == "ECALL"):
                    print(f'ECALL: $finish, LINE: {iterator} (end of program)\n')
                    break

                
        elif(stage == 'R'):
            '''REGISTER'''
            if(d_instruction.name != "N/A" and d_instruction.name != "NOP"):
                (success, err) = register_check(d_instruction, line)
                if(SHOW_REG): reg_file.print()

                if(not success):
                    print(
                        f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<REGISTER> PC: {dec_to_hex(bin_to_dec(d_instruction.pc), 8)}. Instruction: {dec_to_hex(bin_to_dec(d_instruction.binary), 8)} - {get_print_instruction(d_instruction)}, REGISTER FILE was not updated correctly'
                    )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break
            
        elif(stage == 'E'):
            '''EXECUTE'''
            '''
                1. execute(pipeline)
                - give the current pipeline class (for bypassing), update the alu_res and pc_sel
                - return True if no errors

                2. execute_check(instruction, line)
                - compare current instruction properties to those in trace file
                - return Tuple(result: True/False, err: String - error message if False otherwise None)
            '''
            if(x_instruction.name != "N/A"):
                res = execute(pipeline)

                # print("\nIN EXECUTE:")
                # print(x_instruction.id, x_instruction.name, x_instruction.binary, x_instruction.rd, x_instruction.rs1, x_instruction.rs1_data, x_instruction.imm)
                # print()

                success = True
                err = None
                if(x_instruction.name != "NOP" and not is_jump_type(x_instruction.name)):
                    (success, err) = execute_check(x_instruction, line)

                if (not success):
                        print(
                            f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<EXECUTE> PC: {dec_to_hex(bin_to_dec(x_instruction.pc), 8)}. Instruction: {dec_to_hex(bin_to_dec(x_instruction.binary), 8)} - {get_print_instruction(x_instruction)}, did not EXECUTE correctly'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break

        elif(stage == 'M'):
            '''MEMORY'''
            if(m_instruction.name != "N/A"):
                # memory at address alu_res, write data = value at rs2 (for store instructions)
                res = memory(pipeline, dmemory)

                success = True
                err = None
                if(m_instruction.name != "NOP"):
                    (success, err) = memory_check(m_instruction, line, dmemory)

                if (not success):
                        print(
                            f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<MEMORY> PC: {dec_to_hex(bin_to_dec(m_instruction.pc), 8)}. Instruction: {dec_to_hex(bin_to_dec(m_instruction.binary), 8)} - {get_print_instruction(m_instruction)}, MEMORY (dmemory) was not updated correctly'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break

        elif(stage == 'W'):
            '''WRITE'''
            if(w_instruction.name != "N/A"):
                # memory at address rd, write data = value at rs1
                res = write(w_instruction, reg_file)
                
                success = True
                err = None
                if(w_instruction.name != "NOP"):
                    (success, err) = write_check(w_instruction, line)

                if (not success):
                        print(
                            f'<TRACE> ERROR LINE: {iterator} in {TRACE}\n<WRITE> PC: {dec_to_hex(bin_to_dec(w_instruction.pc), 8)}. Instruction: {dec_to_hex(bin_to_dec(w_instruction.binary), 8)} - {get_print_instruction(w_instruction)}, did not WRITE correctly'
                        )
                if(err):
                    print(err)
                if(STOP_ON_ERR and not success):
                    print(f'\n{"Stopping execution..." if STOP_ON_ERR else ""}\n')
                    break

            '''Advance pipeline if needed.'''
            flush_jump = is_jump_type(d_instruction.name)
            br_taken = x_instruction.pc_sel == "1"
            flush_branch = is_branch_type(x_instruction.name) and br_taken

            #// TODO: pipeline.stall()
            # print(f'STALL: {stall}')
            if(stall): 
                # print('===== BEFORE STALL 💩 =====')
                # print(pipeline.print())

                if(SHOW_INSTR):
                    instr_num += 1
                    print(f'{instr_num}. STALL')
                elif(SHOW_INSTR_NO_NUM):
                    instr_num += 1
                    print('STALL')
                # stall = False
                pipeline.stall()

                # print('===== AFTER STALL 💩 =====')
                # print(pipeline.print())
            #// TODO: FIX ID INCREMENT
            else: pipeline.add(Instruction(int(instr_num))) # add blank instruction
            
            '''CLOCK BOUNDARY'''

            '''Flush Instructions if needed'''
            # NOTE: all logic below refer to the state of the system in the previous clock cycle.
            if(flush_jump): 
                # print('===== BEFORE JUMP FLUSH 💩 =====')
                # pipeline.print()
                pipeline.flush_jump()
                # print('===== AFTER JUMP FLUSH 🚽 =====')
                # pipeline.print()
            if(flush_branch): pipeline.flush_branch()
        
        #// TODO: think about how to stall while still advancing the program. (i.e. f.readline())
        # ANS: I think you just keep reading the next line, the expected values should align with the bubble!
            
        line = f.readline()
            

    f.close()


if __name__ == "__main__":
    main()
