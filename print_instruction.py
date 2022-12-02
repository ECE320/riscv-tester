'''
    Pass in a HEX instruction argument
'''


import sys

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






def decode(instruction):
    instruction_name = "N/A"
    opcode = get_opcode(instruction.binary)
    rd = get_rd(instruction.binary)
    rs1 = get_rs1(instruction.binary)
    rs2 = get_rs2(instruction.binary)
    funct3 = get_funct3(instruction.binary)
    funct7 = get_funct7(instruction.binary)
    # imm = immediate_generator(instruction.binary)
    shamt = get_shamt(instruction.binary)
    # rs1_data = reg_file.get(bin_to_dec(rs1))
    # rs2_data = reg_file.get(bin_to_dec(rs2))

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
    # instruction.rs1_data = rs1_data
    # instruction.rs2_data = rs2_data

    imm = immediate_generator(instruction)
    instruction.imm = imm

    if (instruction_name == "JALR"):
        instruction.alu_result = dec_to_bin(bin_to_dec(instruction.imm) + bin_to_dec(instruction.rs1_data), 32)
        # print(f'JALR to {instruction.alu_result}')
    elif (instruction_name == "JAL"):
        instruction.alu_result = dec_to_bin(bin_to_dec(instruction.imm) + bin_to_dec(instruction.pc), 32)

    return True






def get_print_instruction(instruction):
    if(instruction.name == "N/A"): return None
    elif(instruction.name == "ECALL"):
        return "ECALL"

    rd = bin_to_dec(instruction.rd)
    rs1 = bin_to_dec(instruction.rs1)
    rs2 = bin_to_dec(instruction.rs2)
    imm = dec_to_hex(bin_to_dec(instruction.imm), 8)
    shamt = dec_to_hex(bin_to_dec(instruction.rd), 2)

    # index_out = f'{instr_num}. ' if instr_num else ""
    index_out = ""
    print_out = None
    
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


def main():
    if(len(sys.argv)):
        instruction = Instruction()
        instruction.binary = dec_to_bin(hex_to_dec(sys.argv[1]), 32)
        decode(instruction)
        print(get_print_instruction(instruction))


if __name__ == "__main__":
    main()
