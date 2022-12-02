"""
    Authors: Justin Mendes and Shazil Razzaq
    Date: Monday October 17, 2022
    Last Modified: Friday December 2, 2022
    Version: v1.2.1
    
    Runs all of the *.x tests in the benchmarks
    Place this script in the './verif/scripts' 
    of the project you want to test
    i.e. ./projects/pd4/verif/scripts

    USAGE:  python3 test.py [-h/--help] [-e/--execute] [-v/--verify] [any flags/options for parse.py...] <PATH to rv32-benchmarks directory>

    EX:     python3 test.py ../../../../../rv32-benchmarks/

    UPDATES:
    @justincmendes:
    - Newline before each test for better visual separation
    - Ability to run test execution and test verification separately (flags)
    - Support flag args
    -- -e for executing the tests (*.x) only.
    -- -v for verifying the tests (*.trace) only.
"""

import subprocess
import sys
import glob
import argparse
import os

parser = argparse.ArgumentParser()


# https://stackoverflow.com/questions/38834378/path-to-a-directory-as-argparse-argument
# https://stackoverflow.com/a/54547257
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'{path} is not a valid directory')


def main():
    parser.add_argument("path", type=dir_path, help="Path to the rv32-benchmarks folder (ex. ../../../../../rv32-benchmarks) which has subfolders: /individual-instructions/ and /simple-programs/.")
    parser.add_argument("-e", "--execute", dest="execute", action="store_true", help="Add this flag to EXECUTE the tests only (runs all .x tests at the specified path).")
    parser.add_argument("-v", "--verify", dest="verify", action="store_true", help="Add this flag to VERIFY the trace files of the tests only (runs parse.py on all trace files in \"../sim/verilator/test_pd/*.trace\").")

    # Forwarded Flags: parse.py
    parser.add_argument("-i", "--instructions", dest="instructions", action="store_true", help="parse.py: Print all instructions in RISC-V format (Default: False, instructions enumerated)")
    parser.add_argument("-in", "--instructions-no-num", dest="instructions_no_num", action="store_true", help="parse.py: Print all instructions in RISC-V format (Default: False, instructions NOT enumerated)")
    parser.add_argument("-r", "--regfile", dest="regfile", action="store_true", help="parse.py: Print the state of the Register File at each [R] (Default: False)")
    # parser.add_argument("-t", "--terminate", "--ecall", dest="ecall", action="store_true", help="End (terminate) the parse.py check at the first ECALL instruction (Default: False)")
    parser.add_argument("-s", "--skip", dest="skip", action="store_false", help="parse.py: Stop execution on errors (Default: True)")
    parser.add_argument("-m", "--mem", dest="mem", type=int, default=1048576, metavar="MEM_DEPTH", help="parse.py: Number of bytes (8-bit values) in data memory (dmemory). (Default: 1048576)")

    args = parser.parse_args()

    PATH = args.path
    RUN_EXEC = args.execute
    RUN_VERIF = args.verify
    
    # Forwarded Flags: parse.py
    SHOW_INSTR = args.instructions
    SHOW_INSTR_NO_NUM = args.instructions_no_num
    SHOW_REG = args.regfile
    # END_ON_ECALL = args.ecall
    STOP_ON_ERR = args.skip
    MEM_DEPTH = args.mem # # of bytes (8-bit values)


    # Should be covered by positional argument, Default: required (without nargs specified)
    if(not PATH):
        print("Please add a PATH to the rv32-benchmarks folder, (python3 test.py [-h/--help] [-e/--execute] [-v/--verify] <PATH>)\n")
        return

    # If neither are flagged for an exclusive run
    # Default behaviour: Run both.
    if(not RUN_EXEC and not RUN_VERIF):
        RUN_EXEC = True
        RUN_VERIF = True

    # Grabs all the files underneath the path
    # with file extension *.x

    # Execute all .x tests
    if RUN_EXEC:
        test_files = glob.glob(f'{PATH}/individual-instructions/*.x') + glob.glob(f'{PATH}/simple-programs/*.x')

        print("==================\nEXECUTING ALL TESTS\n==================")
        for file in test_files:
            print(f'\n{file}')

            cmd = ["make", "-s", "run", "VERILATOR=1", "TEST=test_pd", f'MEM_PATH={file}']
            
            print(' '.join(cmd))
            subprocess.run(["make", "-s", "run", "VERILATOR=1", "TEST=test_pd", f'MEM_PATH={file}'])

    
    if RUN_VERIF:
        trace_files = glob.glob(f'../sim/verilator/test_pd/*.trace')
        # remove the .trace at the end and only get the name of the file, to search for the test files
        file_names = []
        for file in trace_files:
            filename = os.path.basename(file)
            file_names.append(os.path.splitext(filename)[0])

        test_files = []
        for file in file_names:
            if(file.startswith("rv32ui-p-")):
                test_files.append(f'{PATH}/individual-instructions/{file}.x')
            else:
                test_files.append(f'{PATH}/simple-programs/{file}.x')

        if(RUN_EXEC): print("\n")
        print("==================\nVERIFYING ALL TRACE FILES\n==================")
        for i, file in enumerate(trace_files):
            print(f'\n{file}')

            cmd = ["python3", "parse.py", f'{file}', f'{test_files[i]}']
            
            if SHOW_INSTR: cmd.append("-i")
            if SHOW_INSTR_NO_NUM: cmd.append("-in")
            if SHOW_REG: cmd.append("-r")
            # if END_ON_ECALL: cmd.append("-t")
            if not STOP_ON_ERR: cmd.append("-s")
            if MEM_DEPTH: 
                cmd.append("-m")
                cmd.append(str(MEM_DEPTH))

            print(' '.join(cmd))
            subprocess.run(cmd)

    

if __name__ == "__main__":
    main()
