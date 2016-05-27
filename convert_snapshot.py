#!/usr/bin/env python

import os
import sys
import argparse
import json

def srec_write_s1(addr, data):
    outdata = []
    outdata.append(3 + len(data)) #Byte count
    outdata.append((addr >> 8) & 0xff)
    outdata.append(addr & 0xff)
    outdata += data
    checksum = (~sum(outdata)) & 0xff
    sys.stdout.write("S1%s%02x\n" % ("".join(["%02x" % x for x in outdata]), checksum))
    
def srec_write_s9(addr):
    sys.stdout.write("S903%04x%02x\n" % (addr, ~(3 + (addr >> 8) + (addr & 0xff)) & 0xff))

def main(args, env):
    snapshot = json.load(sys.stdin)
    mem = snapshot["updatememory"]
    lines = map("".join, zip(*[iter(mem)] * 36))
    for line in lines:
        addr = int(line[0:4], 16)
        data = [int("".join(x), 16) for x in zip(*[iter(line[4:])] * 2)]
        
        srec_write_s1(addr, data)
    srec_write_s9(snapshot["regs"][0])
        
        #print("addr = 0x%04x, data = %s" % (addr, " ".join(["%02x" % x for x in data])))

def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    code = main(parse_args(), os.environ)
    if code:
        sys.exit(code)
	
