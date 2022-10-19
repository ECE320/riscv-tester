"""
    Authors: Justin Mendes and Shazil Razzaq
    Date: Thursday October 6, 2022
    Version: v1.0
    
    Runs all of the *.x tests in the benchmarks
    Place this script in the './verif/scripts' 
    of the project you want to test
    i.e. ./projects/pd3/verif/scripts

    USAGE:  python3 test.py <PATH to rv32-benchmarks directory>

    EX:     python3 test.py ../../../../../rv32-benchmarks/
"""

import subprocess
import sys
import glob

def main():
    # Path to the rv32-benchmarks folder
    # ex. ../../../../../rv32-benchmarks/
    PATH = sys.argv[1]

    # Grabs all the files underneath the path
    # with file extension *.x
    test_files = glob.glob(f'{PATH}/individual-instructions/*.x') + glob.glob(f'{PATH}/simple-programs/*.x')

    # Execute all .x tests
    print("==================\nEXECUTING ALL TESTS\n==================")
    for file in test_files:
        print(file)
        subprocess.run(["make", "-s", "run", "VERILATOR=1", "TEST=test_pd", f'MEM_PATH={file}'])

    
    trace_files = glob.glob(f'../sim/verilator/test_pd/*.trace')
    
    print("==================\nVERIFYING ALL TRACE FILES\n==================")
    for file in trace_files:
        print(file)
        subprocess.run(["python3", "parse.py", f'{file}'])

if __name__ == "__main__":
    main()
