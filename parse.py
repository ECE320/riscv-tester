"""
    Authors: Justin Mendes and Shazil Razzaq
    Date: Thursday October 6, 2022
    Version: v1.0

    Tests functionality of RISC-V processor up to PD3

    USAGE:  python3 parse.py <file.trace>

    EX:     python3 parse.py rv32ui-p-sltiu.trace
"""

import sys

class Memory:
    def __init__(self, size, data_width):
        self.arr = ["0" * data_width] * size

    def set(self, str, index):
        self.arr[index] = str
        return True

    def get(self, index):
        return self.arr[index]

def fetch(line):
    instruction_hex = line[13:21]
    pc = line[4:12]
    return (pc, dec_to_bin(hex_to_dec(instruction_hex), 32))

def get_execute_args(line):
    return [line[4:12], line[13:21], line[22]]

def get_opcode(instruction_bin):
    return instruction_bin[-7:-1] + instruction_bin[-1]

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
    # elif(
    #     instruction == "ADD"
    #     or instruction == "SUB"
    #     or instruction == "SLL"
    #     or instruction == "SLT"
    #     or instruction == "SLTU"
    #     or instruction == "XOR"
    #     or instruction == "SRL"
    #     or instruction == "SRA"
    #     or instruction == "OR"
    #     or instruction == "AND"
    #     or instruction == "SLLI"
    #     or instruction == "SRLI"
    #     or instruction == "SRAI"
    # ):
    else:
        imm = sign_extend("0", 0)

    return imm
    
def get_shamt(instruction_bin):
    return instruction_bin[-12:-7]

def signed_less_than(op1, op2):
    # Positive < Negative
    if op1[0] < op2[0]:
        res = False
    # Negative < Positive
    elif(op2[0] < op1[0]):
        res = True
    # Same sign
    else:
        if(not op1[0]):
            res = bin_to_dec(op1) < bin_to_dec(op2)
        else:
            res = bin_to_dec(op2) < bin_to_dec(op1)
    return res

def update_reg_file(instruction_bin, exec_res, pc, reg_file):
    instruction = decoder(instruction_bin)
    # opcode = get_opcode(instruction_bin)
    rd_addr = bin_to_dec(get_rd(instruction_bin))
    rd = reg_file.get(bin_to_dec(get_rd(instruction_bin)))
    rs1 = reg_file.get(bin_to_dec(get_rs1(instruction_bin)))
    rs2 = reg_file.get(bin_to_dec(get_rs2(instruction_bin)))
    funct3 = get_funct3(instruction_bin)
    funct7 = get_funct7(instruction_bin)
    imm = get_imm(instruction_bin)
    shamt = get_shamt(instruction_bin)

    res = False

    alu_res = exec_res[0]
    br_taken = exec_res[1]

    # WB: PC+4
    if(
        instruction == "JAL"
        or instruction == "JALR"
    ):
        res = reg_file.set(dec_to_bin(bin_to_dec(pc) + 4), rd_addr)

    # WB: MEM
    elif (
        instruction == "LB"
        or instruction == "LH"
        or instruction == "LW"
        or instruction == "LBU"
        or instruction == "LHU"
    ): 
        # Get the word from memory
        # mem_res = None
        
        # Byte:
        # res = reg_file.set(alu_res[-8:-1], rd_addr)

        # Half Word:
        # res = reg_file.set(alu_res[-16:-1], rd_addr)

        # Word:
        # res = reg_file.set(alu_res, rd_addr)
        None
    
    # WB: ALU
    elif (
        instruction == "ADDI"
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
        res = reg_file.set(alu_res, rd_addr)

    return res


def execute(instruction_bin, pc, reg_file):
    instruction = decoder(instruction_bin)
    # opcode = get_opcode(instruction_bin)
    rd = reg_file.get(bin_to_dec(get_rd(instruction_bin)))
    rs1 = reg_file.get(bin_to_dec(get_rs1(instruction_bin)))
    rs2 = reg_file.get(bin_to_dec(get_rs2(instruction_bin)))
    funct3 = get_funct3(instruction_bin)
    funct7 = get_funct7(instruction_bin)
    imm = get_imm(instruction_bin)
    shamt = get_shamt(instruction_bin)
    
    alu_res = "0" * 32
    br_taken = "0"

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
        # print("EXECUTE:")
        # print(bin_to_dec(rs1))
        # print(imm)
        # print(bin_to_dec(imm))
        # print(dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32))
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
        alu_res = rs1[0] + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 31)


    # R-TYPES
    elif (instruction == "ADD"):
        alu_res = dec_to_bin(bin_to_dec(rs1) + bin_to_dec(rs2), 32)

    elif (instruction == "SUB"):
        alu_res = dec_to_bin(bin_to_dec(rs1) - bin_to_dec(rs2), 32)

    elif (instruction == "SLL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) << bin_to_dec(rs2), 32)

    elif (instruction == "SLT"):
        alu_res = "0" * 31 + ("1" if signed_less_than(rs1, rs2) else "0")

    elif (instruction == "SLTU"):
        alu_res = "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0")

    elif (instruction == "XOR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(rs2), 32)

    elif (instruction == "SRL"):
        alu_res = dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2), 32)

    elif (instruction == "SRA"):
        alu_res = rs1[0] + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2), 31)

    elif (instruction == "OR"):
        alu_res = dec_to_bin(bin_to_dec(rs1) | bin_to_dec(rs2), 32)

    elif (instruction == "AND"):
        alu_res = dec_to_bin(bin_to_dec(rs1) & bin_to_dec(rs2), 32)

    # elif (instruction == "ECALL"):

    return (alu_res, br_taken)

def execute_check(line, instruction_bin, reg_file):
    instruction = decoder(instruction_bin)
    # opcode = get_opcode(instruction_bin)
    # rd = reg_file.get(bin_to_dec(get_rd(instruction_bin)))
    rs1 = reg_file.get(bin_to_dec(get_rs1(instruction_bin)))
    # print(rs1)
    rs2 = reg_file.get(bin_to_dec(get_rs2(instruction_bin)))
    # print(rs2)
    # funct3 = get_funct3(instruction_bin)
    # funct7 = get_funct7(instruction_bin)
    imm = get_imm(instruction_bin)
    # print(f'instruction: {instruction}, imm: {imm}')
    shamt = get_shamt(instruction_bin)

    exec_args = get_execute_args(line)
    pc = dec_to_bin(hex_to_dec(exec_args[0]), 32)
    # print(exec_args)
    # print(hex_to_dec(exec_args[1]))
    # print(dec_to_bin(hex_to_dec(exec_args[1]), 32))
    alu_res = dec_to_bin(hex_to_dec(exec_args[1]), 32)
    br_taken = dec_to_bin(hex_to_dec(exec_args[2]), 1)
    # print(pc)

    res = 0

    # U-TYPES
    if (instruction == "LUI"): res = alu_res == zero_extend(imm[-32:-12], 11)
    
    # Add IMM to PC
    elif (
        instruction == "AUIPC"
        or instruction == "JAL"
    ): 
        res = alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32)
        
    # B-TYPES
    elif (instruction == "BEQ"):
        res = (br_taken == ("1" if rs1 == rs2 else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))
    elif (instruction == "BNE"):
        res = (br_taken != ("1" if rs1 == rs2 else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))
    elif (instruction == "BLT"):
        res = (br_taken == ("1" if signed_less_than(rs1, rs2) else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))
    elif (instruction == "BGE"):
        res = (br_taken != ("1" if signed_less_than(rs1, rs2) else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))
    elif (instruction == "BLTU"):
        res = (br_taken == ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))
    elif (instruction == "BGEU"):
        res = (br_taken != ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0") and alu_res == dec_to_bin(bin_to_dec(pc) + bin_to_dec(imm), 32))


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
        # print("EXECUTE_CHECK:")
        # print(bin_to_dec(rs1))
        # print(imm)
        # print(bin_to_dec(imm))
        # print(alu_res)
        # print(dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32))
        res = alu_res == dec_to_bin(bin_to_dec(rs1) + bin_to_dec(imm), 32)

    elif (instruction == "SLTI"):
        res = alu_res == "0" * 31 + ("1" if signed_less_than(rs1, imm) else "0")

    elif (instruction == "SLTIU"):
        res = alu_res == "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(imm) else "0")

    elif (instruction == "XORI"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(imm), 32)

    elif (instruction == "ORI"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) | bin_to_dec(imm), 32)

    elif (instruction == "ANDI"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) & bin_to_dec(imm), 32)

    elif (instruction == "SLLI"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) << bin_to_dec(shamt), 32)

    elif (instruction == "SRLI"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 32)

    elif (instruction == "SRAI"):
        res = alu_res == rs1[0] + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(shamt), 31)

    # R-TYPES
    elif (instruction == "ADD"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) + bin_to_dec(rs2), 32)

    elif (instruction == "SUB"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) - bin_to_dec(rs2), 32)

    elif (instruction == "SLL"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) << bin_to_dec(rs2), 32)

    elif (instruction == "SLT"):
        res = alu_res == "0" * 31 + ("1" if signed_less_than(rs1, rs2) else "0")

    elif (instruction == "SLTU"):
        res = alu_res == "0" * 31 + ("1" if bin_to_dec(rs1) < bin_to_dec(rs2) else "0")

    elif (instruction == "XOR"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) ^ bin_to_dec(rs2), 32)

    elif (instruction == "SRL"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2), 32)

    elif (instruction == "SRA"):
        res = alu_res == rs1[0] + dec_to_bin(bin_to_dec(rs1) >> bin_to_dec(rs2), 31)

    elif (instruction == "OR"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) | bin_to_dec(rs2), 32)

    elif (instruction == "AND"):
        res = alu_res == dec_to_bin(bin_to_dec(rs1) & bin_to_dec(rs2), 32)

        
    elif (instruction == "ECALL"):
        res = True

    return res

def get_decoder_args(line):
    values = line[12:41]
    args = values.split()
    return args

def decoder(instruction_bin):
    instruction = "N/A"
    opcode = instruction_bin[-7:-1] + instruction_bin[-1]
    # print(f'DECODER_INSTR_BIN: {instruction_bin}')
    # print(f'DECODER_OPCODE: {opcode}')
    funct3 = instruction_bin[-15:-12]
    # print(f'FUNCT_3: {funct3}')
    funct7 = instruction_bin[-32:-25]
    # print(f'FUNCT_7: {funct7}')

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

    

# bin = immediate value
# sign_bit_index = sign bit index in the imm[]
def sign_extend(bin, sign_bit_index, size = 32):
    return bin[-(sign_bit_index + 1)] * (size - 1 - sign_bit_index) + bin

# Extends down
def zero_extend(bin, start_index):
    return bin + "0" * (start_index + 1)
    

def decoder_check(line, instruction_bin):
    args = get_decoder_args(line)
    instruction = decoder(instruction_bin)

    # [D] 01000000 [13 01 00 00 0 00 00000000 00]
    opcode = dec_to_bin(hex_to_dec(args[0]), 7)
    rd = dec_to_bin(hex_to_dec(args[1]), 5)
    rs1 = dec_to_bin(hex_to_dec(args[2]), 5)
    rs2 = dec_to_bin(hex_to_dec(args[3]), 5)
    funct3 = dec_to_bin(hex_to_dec(args[4]), 3)
    funct7 = dec_to_bin(hex_to_dec(args[5]), 7)
    imm = dec_to_bin(hex_to_dec(args[6]), 32)
    shamt = dec_to_bin(hex_to_dec(args[7]), 5)

    res = False

    # U-TYPES
    if (
        instruction == "LUI"
        or instruction == "AUIPC"
    ):        
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-12:-7] == rd) and (sign_extend(zero_extend(instruction_bin[-32:-12], 11), 31) == imm)) else False

    # J-TYPES
    elif (instruction == "JAL"):        
        temp_string = instruction_bin[-32] + instruction_bin[-20:-12] + instruction_bin[-21] + instruction_bin[-31:-21] #Check
        # print(temp_string)

        # print((instruction_bin[-7:-1] + instruction_bin[-1] == opcode))
        # print((instruction_bin[-12:-7] == rd))

        # print(instruction_bin[-21:-1] + instruction_bin[-1])
        # print(zero_extend(instruction_bin[-21:-1] + instruction_bin[-1], 0))
        # print((sign_extend(zero_extend(instruction_bin[-21:-1] + instruction_bin[-1], 0), 20)))
        # print(imm)
        # print((sign_extend(zero_extend(instruction_bin[-21:-1] + instruction_bin[-1], 0), 20) == imm))
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-12:-7] == rd) and (sign_extend(zero_extend(temp_string, 0), 20) == imm)) else False
        
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
        # print("!!!HERE!!!")
        # print(instruction_bin[-7:-1] + instruction_bin[-1])
        # print(opcode)
        # print((instruction_bin[-7:-1] + instruction_bin[-1] == opcode))
        # print((instruction_bin[-12:-7] == rd) and (instruction_bin[-15:-12] == funct3))
        # print((instruction_bin[-20:-15] == rs1))

        # print(instruction_bin[-32:-20])
        # print((sign_extend(instruction_bin[-32:-20], 11)))
        # print(imm)
        # print((sign_extend(instruction_bin[-32:-20], 11) == imm))
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-12:-7] == rd) and (instruction_bin[-15:-12] == funct3) and (instruction_bin[-20:-15] == rs1) and (sign_extend(instruction_bin[-32:-20], 11) == imm)) else False
    
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
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-15:-12] == funct3) and (instruction_bin[-20:-15] == rs1) and (sign_extend(zero_extend(temp_string, 0), 12) == imm)) else False

    # S-TYPES
    elif(
        instruction == "SB"
        or instruction == "SH"
        or instruction == "SW"
    ):
        temp_string = instruction_bin[-32:-25] + instruction_bin[-12:-7]
        # print(temp_string)
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-15:-12] == funct3) and (instruction_bin[-20:-15] == rs1) and (instruction_bin[-25:-20] == rs2) and(sign_extend(temp_string, 11) == imm)) else False
        
    # R-TYPES with SHAMT
    elif(
        instruction == "SLLI"
        or instruction == "SRLI"
        or instruction == "SRAI"
    ):
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-12:-7] == rd) and (instruction_bin[-15:-12] == funct3) and (instruction_bin[-20:-15] == rs1) and (instruction_bin[-25:-20] == shamt) and (instruction_bin[-32:-25] == funct7)) else False
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
    ):
        res = True if ((instruction_bin[-7:-1] + instruction_bin[-1] == opcode) and (instruction_bin[-12:-7] == rd) and (instruction_bin[-15:-12] == funct3) and (instruction_bin[-20:-15] == rs1) and (instruction_bin[-25:-20] == rs2) and (instruction_bin[-32:-25] == funct7)) else False
    
    elif (instruction == "ECALL"):
        res = True if (instruction_bin[-32:-7] == "0" * 25) else False
    
    return res


def hex_to_dec(hex):
    return int(hex, 16)

def dec_to_bin(dec, size):
    bin = format(dec, f'#0{size + 2}b')
    bin = bin[2:]
    return bin[-size:-1] + bin[-1]

def bin_to_dec(bin):
    return int(bin, 2)

def dec_to_hex(dec, size):
    hex = format(dec, f'#0{size + 2}x')
    hex = hex[2:]
    return hex[-size:-1] + hex[-1]



def main():
    PATH = sys.argv[1]
    f = open(PATH, 'r')
    line = f.readline()

    reg_file = Memory(32, 32)
    instruction_bin = ""
    instruction = ""
    pc = ""
    correct = False
    dec_args = []
    exec_args = []

    iterator = 0

    while (line):
        iterator += 1
        # print(line)
        if(line[1] == 'F'):
            (pc, instruction_bin) = fetch(line)
            # print(f'INSTR BINARY:{instruction_bin}')
                
        elif(line[1] == 'D'):
            if(instruction_bin != "0" * 32):
                instruction = decoder(instruction_bin)
                # print(f'INSTRUCTION: {instruction}')
                dec_args = get_decoder_args(line)
                correct = decoder_check(line, instruction_bin)
                if(not correct and instruction != "N/A"):
                    print(f'<DECODER> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, did not DECODE correctly.')

        elif(line[1] == 'R'):
            # print("R")
            None

        elif(line[1] == 'E'):
            if(instruction_bin != "0" * 32):
                correct = execute_check(line, instruction_bin, reg_file)
                if (not correct and instruction != "N/A"):
                        print(f'<EXECUTE> Instruction: {dec_to_hex(bin_to_dec(instruction_bin), 8)}, {instruction}, did not EXECUTE correctly.')
                exec_args = execute(instruction_bin, dec_to_bin(hex_to_dec(pc), 32), reg_file)
                # print(f'{instruction}: {exec_args}')
    
        line = f.readline()


if __name__ == "__main__":
    main()
