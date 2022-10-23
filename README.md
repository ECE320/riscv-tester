# Overview

**Authors: Justin Mendes and Shazil Razzaq**

Tests the functionality of your RV32UI processor by executing test_pd with Verilator and verifying the results with a mock RV32UI processor created in Python.

1. Runs all of the `*.x` tests in the benchmarks (rv32-benchmarks).

2. Verifies all of the resulting `*.trace` files in `../sim/verilator/test_pd/` for behaviour

# TLDR;

1. Put `test.py` and `parse.py` in **`/projects/pd4/verif/scripts`**

2. In `/project/pd4/verif/scripts` enter `python3 test.py <PATH to /rv32-benchmarks>`


<img width="560" alt="image" src="https://user-images.githubusercontent.com/50083088/196791253-b04d6503-5fe4-404c-a53e-4e4c0aa3e5cf.png">

If your output looks like this, you're processor is likely good to go!

Otherwise, read on for debugging instructions.


# Setup/Configuration

1. Place `test.py` and `parse.py` in the **`./verif/scripts`** directory of the project you want to test (e.g. ece320/projects/pd4/verif/scripts).

2. Have local access to `/rv32-benchmarks` found in the ece320 GitLab.

For an end-to-end test follow instructions for [test.py](#test.py).

For testing individual programs follow instructions for [parse.py](#parse.py)


# test.py

Runs a full end-to-end test of your processor.

**Default Behaviour:** Executes and Verifies all tests.

**Assumptions Made**: 
1. Your `/rv32-benchmarks` directory has subfolders: `/individual-instructions` and `/simple-programs`.

2. Your results - `*.trace` files, are in directory `../sim/verilator/test_pd`

## Format

`python3 test.py [-h/--help] [-e/--execute] [-v/--verify] [any flags/options for parse.py...] <PATH: to "/rv32-benchmarks" directory>`

***Must be ran in `/projects/pd#/verif/scripts/` directory***

### Arguments

`PATH`: **Required.** 

Path to "/rv32-benchmarks" directory.

### Flags

Optional flags to help with debugging:

1. `-h`/`--help`: 

Get more information how to run the command and what each of the arguments and options/flags do.


2. `-e`/`--execute`:

Add this flag to EXECUTE the tests **only** (runs all .x tests at the specified path).


3. `-v`/`--verify`: 

Add this flag to VERIFY the trace files of the tests **only** (runs [parse.py](#parse.py) on all trace files in \"../sim/verilator/test_pd/*.trace\").


### Other Flags (Forwarded)

Flags forwarded to [parse.py](##format-1):

1. `-s`/`--skip`
2. `-i`/`--instructions`
3. `-in`/`--instructions-no-mem`
4. `-r`/`--regfile`
5. `-m <MEM_DEPTH>`


## Examples

`python3 test.py ../../../../../rv32-benchmarks -in -s`

`python3 test.py -e ../../../../../rv32-benchmarks`

`python3 test.py -i -r ../../../../../rv32-benchmarks --verify`

`python3 test.py -v --skip -m 1024000 /home/ece320/rv32-benchmarks -i`

`python3 test.py --execute ../../../../../rv32-benchmarks`

`python3 test.py ../../../../../rv32-benchmarks --execute`



# parse.py

Compares the results of your processor's `*.trace` output file under a `*.x` testbench to a mock Python RV32UI processor to verify correctness of your processor under the given `*.x` test.

## Format

`python3 parse.py [-h/--help] [-s/--skip] [-i/--instructions] [-in/--instructions-no-num] [-r/--regfile] [-m <MEM_DEPTH> / --mem <MEM_DEPTH>] <PATH: to .trace file> `

### Arguments

`PATH`: **Required.** 

Path to .trace file from executing a `*.x` testbench: `make run VERILATOR=1 TEST=test_pd MEM_PATH=*.x`.

### Flags

1. `-s`/`--skip`: Default **False**

CONTINUE execution on errors.

2. `-i`/`--instructions`: Default: **False**

Print all instructions in RISC-V format (instructions enumerated).

3. `-in`/`--instructions-no-num`: Default: **False**

Print all instructions in RISC-V format (instructions NOT enumerated).

4. `-r`/`--regfile`: Default: **False**

Print the state of the Register File at each [R] in the .trace file.

5. `-m <MEM_DEPTH>`/`--mem <MEM_DEPTH>`: Default: **-m 1048576**

Number of bytes (8-bit values) in data memory (memory).


## Examples

`python3 parse.py ../sim/verilator/test_pd/rv32ui-p-sltiu.trace -in -s`

`python3 parse.py -i -r ../data/rv32ui-p-add.trace`

`python3 parse.py --skip -m 1024000 ../tests/out/BubbleSort.trace -i`


# Future Versions

## v1.2

1. Add support for flags `-pd1` `-pd2` `-pd3` `-pd4` `-pd5` to allow user to isolate which project they would like to test up to.
