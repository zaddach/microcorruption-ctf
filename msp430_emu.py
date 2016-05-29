#!/usr/bin/env python3

import os
import sys
import argparse
import json

from enum import Enum

def sext(val, bits):
    """Extend a value with @bits bits to a signed 16 bit value"""
    if val & (1 << (bits - 1)):
        return ((0xffff << (bits - 1)) | val) & 0xffff
    else:
        return val
        
class GdbProtocolState(Enum):
    NONE = 0
    BODY = 1
    CHECKSUM_1 = 2
    CHECKSUM_2 = 3
        
class GdbServer():
    def __init__(self, connection, cpu):
        self.connection = connection
        self.cpu = cpu
        
    def start(self):
        pass
        
    def _parse_packet(self, packet, checksum):
        pass
        
    def run(self):
        packet = None
        checksum = None
        state = GdbProtocolState.NONE
        while True:
            byte = self.connection.read(1)
            if byte == ord('$'):
                state = GdbProtocolState.BODY
                packet = []
            elif byte == ord('#'):
                state = gdbProtocolState.CHECKSUM_1
            else:
                if state == GdbProtocolState.BODY:
                    packet.append(byte)
                elif state == GdbProtocolState.CHECKSUM_1:
                    checksum = int(byte, 16) * 16
                elif state == GdbProtocolState.CHECKSUM_2:
                    checksum += int(byte, 16)
                    if self._parse_packet("".join(packet), checksum):
                        self.connection.write("+")
                    else:
                        self.connection.write("-")
                    self.connection.flush()
                    state = GdbProtocolState.NONE
                    packet = None
        
    
        

class MSP430Instruction():
    def __init__(self, len):
        self.len = len
        
    def eval(self, cpu):
        cpu.set_pc(cpu.get_pc() + self.len)
        
    def _set_flags(self, cpu, flags, val, dst):
        if 'C' in flags:
            cpu.set_flag(CPUFlags.C, val > 0xff if self.size == 1 else val > 0xffff)
        if 'V' in flags:
            dst_N = (dst >> (0x80 if self.size == 1 else 0x8000)) & 1 != 0
            val_N = (val >> (0x80 if self.size == 1 else 0x8000)) & 1 != 0
            cpu.set_flag(CPUFlags.V, dst_N ^ val_N)
        if 'N' in flags:
            cpu.set_flag(CPUFlags.N, val & (1 << (7 if self.size == 1 else 15)) != 0)
        if 'Z' in flags:
            cpu.set_flag(CPUFlags.Z, val & (0xff if self.size == 1 else 0xffff) == 0)

class Condition(Enum):
    not_equal = 0
    not_zero = 0
    equal = 1
    zero = 1
    no_carry = 2
    lower = 2
    carry = 3
    higher_same = 3
    negative = 4
    greater_equal = 5
    less = 6
    always = 7

class JumpInst(MSP430Instruction):
    def __init__(self, offset, condition = Condition.always):
        self.offset = offset
        self.condition = condition
        super().__init__(2)
        
    def eval(self, cpu):
        cond = {
            Condition.not_equal: lambda cpu: not cpu.get_flag(CPUFlags.Z),
            Condition.equal: lambda sr: cpu.get_flag(CPUFlags.Z),
            Condition.no_carry: lambda sr: not cpu.get_flag(CPUFlags.C),
            Condition.carry: lambda sr: cpu.get_flag(CPUFlags.C),
            Condition.negative: lambda sr: cpu.get_flag(CPUFlags.N),
            Condition.greater_equal: lambda sr: cpu.get_flag(CPUFlags.N) == cpu.get_flag(CPUFlags.V),
            Condition.less: lambda sr: cpu.get_flag(CPUFlags.N) == cpu.get_flag(CPUFlags.V),
            Condition.always: lambda sr: True}[self.condition](cpu)
            
        if cond:
            cpu.set_pc((cpu.get_pc() + self.offset) & 0xffff)
        else:
            super().eval(cpu)
        
    def __str__(self):
        mnem = {
            Condition.not_equal:     "jne",
            Condition.equal:         "jeq",
            Condition.no_carry:      "jnc",
            Condition.carry:         "jc",
            Condition.negative:      "jn",
            Condition.greater_equal: "jge",
            Condition.less:          "jl",
            Condition.always:        "jmp"}
            
        offset = self.offset
        sign = "+"
        if offset >= 0x8000:
           offset = 0x10000 - offset
           sign = "-"
            
        return "%s .%s0x%x" % (mnem[self.condition], sign, offset)
        
class Operand(object):
    pass
    
    def _regname(self, reg):
        if reg == 0:
            return "pc"
        elif reg == 1:
            return "sp"
        elif reg == 2:
            return "sr"
        else:
            return "r%d" % reg
        
class SourceOperand(Operand):
    pass
    
class DummyOperand(Operand):
    def __init__(self, len = 0):
        self.len = len
        
    def get(self, cpu):
        return 0
        
    def set(self, cpu, val):
        pass
        
    def __str__(self):
        return "<dummy>"
    
class RegisterOperand(Operand):
    def __init__(self, reg, size):
        self.reg = reg
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        mask = 0xff if self.size == 1 else 0xffff
        return cpu.get_register(self.reg) & mask
        
    def set(self, cpu, val):
        mask = 0xff if self.size == 1 else 0xffff
        cpu.set_register(self.reg, val & mask)
        
    def __str__(self):
        return self._regname(self.reg)
            
class AbsoluteOperand(Operand):
    def __init__(self, address, size):
        self.address = address
        self.size = size
        self.len = 2
        
    def get(self, cpu):
        return cpu.memory.read(self.address, self.size)
        
    def set(self, cpu, val):
        cpu.memory.write(self.address, self.size, val)
        
    def __str__(self):
        return "&%04x" % self.address
        
class ConstantOperand(Operand):
    def __init__(self, val, size):
        self.val = val
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        return self.val
        
    def __str__(self):
        return "#%x" % self.val
            
class RegisterOffsetOperand(Operand):
    def __init__(self, reg, offset, size):
        self.reg = reg
        self.offset = offset
        self.size = size
        self.len = 2
        
    def get(self, cpu):
        return cpu.memory.read(cpu.get_register(self.reg) + self.offset, self.size)
        
    def set(self, cpu, val):
        cpu.memory.write(cpu.get_register(self.reg) + self.offset, self.size, val)
        
    def __str__(self):
        return "%x(%s)" % (self.offset, self._regname(self.reg))
        
class RegisterIndirectOperand(Operand):
    def __init__(self, reg, size):
        self.reg = reg
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        return cpu.memory.read(cpu.get_register(self.reg), self.size)
        
    def __str__(self):
        return "%x(%s)" % (self.offset, self._regname(self.reg))
        
class RegisterIndirectIncrementOperand(Operand):
    def __init__(self, reg, size):
        self.reg = reg
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        value = cpu.memory.read(cpu.get_register(self.reg), self.size)
        cpu.set_register(self.reg, cpu.get_register(self.reg) + self.size)
        return value
        
    def __str__(self):
        return "@%s+" % (self._regname(self.reg), )
    
class ImmediateOperand(Operand):
    def __init__(self, val, size):
        self.val = val
        self.size = size
        self.len = 2
        
    def get(self, cpu):
        mask = 0xff if self.size == 1 else 0xffff
        return self.val & mask
        
    def __str__(self):
        return "#%04x" % self.val
        
class UnaryInst(MSP430Instruction):
    def __init__(self, op, size):
        self.op = op
        self.size = size
        self.len = 2 + op.len
        super().__init__(2 + op.len)
        
    def eval(self, cpu):
        super().eval(cpu)
        
    def __str__(self):
        mnem = self.mnem + (".b" if self.size == 1 else "")
        return "%s %s" % (mnem, str(self.op))
        
class RRAInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "rra"
        super().__init__(op, size)
        
    def eval(self, cpu):
        val = op.get()
        msb = val & (0x80 if self.size == 1 else 0x8000)
        result = (val >> 1) | msb
        cpu.set_flag(CPUFlags.C, val & 1 != 0)
        cpu.set_flag(CPUFlags.N, msb != 0)
        cpu.set_flag(CPUFlags.Z, result == 0)
        cpu.set_flag(CPUFlags.V, False)
        self.op.set(cpu, result)
        super().eval(cpu)
        
class RRCInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "rrc"
        super().__init__(op, size)
        
    def eval(self, cpu):
        val = self.op.get(cpu)
        if cpu.get_flag(CPUFlags.C):
            old_c = 0x80 if self.size == 1 else 0x8000
        else:
            old_c = 0
        result = old_c | (val >> 1)
        cpu.set_flag(CPUFlags.C, val & 1 != 0)
        cpu.set_flag(CPUFlags.N, old_c != 0)
        cpu.set_flag(CPUFlags.Z, result == 0)
        cpu.set_flag(CPUFlags.V, False)
        self.op.set(cpu, result)
        super().eval(cpu)
  
class PushInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "push"
        super().__init__(op, size)
        
    def eval(self, cpu):
        cpu.set_sp(cpu.get_sp() - 2)
        cpu.memory.write(cpu.get_sp(), self.size, self.op.get(cpu))
        super().eval(cpu)
        
class CallInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "call"
        super().__init__(op, size)
        
    def eval(self, cpu):
        cpu.set_sp(cpu.get_sp() - 2)
        cpu.memory.write(cpu.get_sp(), self.size, cpu.get_pc() + self.len)
        cpu.set_pc(self.op.get(cpu))
        
class SwpbInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "swpb"
        super().__init__(op, size)
        
    def eval(self, cpu):
        val = self.op.get(cpu)
        self.op.set(cpu, (val >> 8) | ((val & 0xff) << 8))
        super().eval(cpu)
      
        
class BinaryInst(MSP430Instruction):
    def __init__(self, source, destination, size):
        self.source = source
        self.destination = destination
        self.size = size
        super().__init__(2 + source.len + destination.len)
        
    def eval(self, cpu):
        #Use lambda expressions so that operand is only fetched if it is really used
        src = lambda: self.source.get(cpu)
        dst = lambda: self.destination.get(cpu)
        (result, flags) = self._eval(cpu, src, dst)
        if result != None:
            self.destination.set(cpu, result)
        self._set_flags(cpu, flags, result, dst())
        if not (isinstance(self.destination, RegisterOperand) and self.destination.reg == 0):
            #Only advance program counter if it hasn't been modified by the instruction
            super().eval(cpu)
        
    def __str__(self):
        mnem = self.mnem + (".b" if self.size == 1 else "")
        return "%s %s, %s" % (mnem, str(self.source), str(self.destination))
        
class MovInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "mov"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src(), "")
        
class AndInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "and"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        result = src() & dst()
        cpu.set_flag(CPUFlags.V, False)
        cpu.set_flag(CPUFlags.C, result != 0)
        return (result, "NZ")
        
class BisInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "bis"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src() | dst(), "")
        
class CmpInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "cmp"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        result = dst() - src()
        self._set_flags(cpu, "CVNZ", (~src() & (0xff if self.size == 1 else 0xffff)) + 1 + dst(), dst())
        return (None, "")
        
class SubInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "sub"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return ((~src() & (0xff if self.size == 1 else 0xffff)) + 1 + dst(), "CVNZ")

class AddInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "add"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src() + dst(), "CVNZ")
        
class DaddInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "dadd"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        raise RuntimeError("not implemented")
        return (src() + dst(), "CVNZ")

class Memory():
    def __init__(self):
        self.data = [0] * 0x10000
        
    def read(self, address, size):
        return {
            1: self.data[address & 0xffff],
            2: self.data[address & 0xffff] | (self.data[(address + 1) & 0xffff] << 8)
        }[size]

    def write(self, address, size, val):
        if size == 1:
            self.data[address & 0xffff] = val & 0xff
        elif size == 2:
            self.data[address & 0xffff] = val & 0xff
            self.data[(address + 1) & 0xffff] = (val >> 8) & 0xff
        else:
            assert(False)
            
class CPUFlags(Enum):
    C = 0
    Z = 1
    N = 2
    V = 8
    GIE = 3
    CPUOFF = 4

class MSP430Cpu():
    def __init__(self, memory, startaddr = 0x4400):
        self.memory = memory
        self.registers = [startaddr] + [0] * 15
        self.memory.write(0x10, 2, 0x4130)
        self.rand = [0x1d57]
        
    def step(self, verbose = False):
        if self.get_pc() == 0x10 and self.get_sr() & 0x8000 != 0: #This is the call gate
            self._handle_callgate()
        elif self.get_sr() & (1 << CPUFlags.CPUOFF.value) != 0:
            print("CPUOFF bit set. Terminating.")
            sys.exit(0)
        else:
            inst = self.decode_instruction(self.get_pc())
            if verbose:
                print("%04x: %s" % (self.get_pc(), str(inst)))
            inst.eval(self)
            
    def _handle_callgate(self):
        gate_num = (self.get_sr() >> 8) & 0x7f
        if gate_num == 0:
            char = self.memory.read(self.get_sp() + 8, 2)
            sys.stdout.write(chr(char))
        elif gate_num == 1:
            self.set_register(15, ord(sys.stdin.read(1)))
        elif gate_num == 2:
            buf_addr = self.memory.read(self.get_sp() + 8, 2)
            buf_size = self.memory.read(self.get_sp() + 10, 2)
            
            sys.stderr.write("INPUT> ")
            sys.stderr.flush()
            
            line = sys.stdin.readline().strip()
            if line.startswith(":"):
                data = [int("".join(x), 16) for x in zip(*[iter(line[1:])] * 2)]
            else:
                data = [ord(x) for x in line]
            for (byte, i) in zip(data, range(buf_size)):
                self.memory.write(buf_addr + i, 1, byte)
        elif gate_num == 0x20:
            self.set_register(15, self.rand.pop())
        elif gate_num == 0x7f:
            sys.stderr.write("Deadbolt unlocked. You're done.")
            sys.exit(0)
        else:
            print("Unhandled callgate code: 0x%02x" % gate_num)
            assert(False)
        
        self.set_pc(self.memory.read(self.get_sp(), 2))
        self.set_sp(self.get_sp() + 2)
                
            
        
    def _decode_source(self, address, As, reg, size):
        if reg == 0:
            return {
                0: RegisterOperand(reg, size),
                1: RegisterOffsetOperand(reg, self.memory.read(address, 2), size),
                3: ImmediateOperand(self.memory.read(address, size), size)}[As]
            assert(False)
        elif reg == 2:
            return {
                0: RegisterOperand(reg, size),
                1: AbsoluteOperand(self.memory.read(address, 2), size),
                2: ConstantOperand(4, size),
                3: ConstantOperand(8, size)}[As]
        elif reg == 3:
            return {
                0: ConstantOperand(0, size),
                1: ConstantOperand(1, size),
                2: ConstantOperand(2, size),
                3: ConstantOperand(0xffff, size)}[As]
            assert(False)
        else:
            return {
                0: RegisterOperand(reg, size),
                1: RegisterOffsetOperand(reg, self.memory.read(address, 2), size),
                2: RegisterIndirectOperand(reg, size),
                3: RegisterIndirectIncrementOperand(reg, size)}[As]
    
    def _decode_destination(self, address, Ad, reg, size):
        if reg == 0:
            return {
                0: RegisterOperand(reg, size)
            }[Ad]
        elif reg == 2:
            return {
                0: RegisterOperand(reg, size),
                1: AbsoluteOperand(self.memory.read(address, size), size)
            }[Ad]
        elif reg == 3:
            return DummyOperand()
        else:
            return {
                0: RegisterOperand(reg, size),
                1: RegisterOffsetOperand(reg, self.memory.read(address, 2), size)}[Ad]
        
    def decode_instruction(self, address):
        # see http://www.ece.utep.edu/courses/web3376/Links_files/MSP430%20Quick%20Reference.pdf
        # and http://www.ti.com/lit/ug/slau144j/slau144j.pdf
        opcode = self.memory.read(address, 2)
        print("opcode = %04x" % opcode)
        if opcode & 0xe000 == 0x2000:
            return JumpInst((sext(opcode & 0x3ff, 10) * 2 + 2) & 0xffff, Condition((opcode >> 10) & 7))
        elif opcode & 0xfc00 == 0x1000:
            opc = (opcode >> 7) & 0x7
            size = 1 if (opc & 1 == 0) and ((opcode >> 6) & 1 != 0) else 2
            if opc == 6:
                assert(False)
            else:
                As = (opcode >> 4) & 3
                src = opcode & 0xf
                size = 1 if (opcode >> 6) & 1 != 0 else 2
                opc = (opcode >> 7) & 0x7
                op = self._decode_source(address + 2, As, src, size)
                return {
                    0x0: RRCInst,
                    0x1: SwpbInst,
                    0x2: RRAInst,
                    0x4: PushInst,
                    0x5: CallInst,
                }[opc](op, size)
                #return RetiInst()
            assert(False)
        else:
            opc = (opcode >> 12) & 0xf
            size = 1 if (opcode >> 6) & 1 != 0 else 2
            As = (opcode >> 4) & 3
            Ad = (opcode >> 7) & 1
            src = (opcode >> 8) & 0xf
            dst = opcode & 0xf
            
            src_op = self._decode_source(address + 2, As, src, size)
            dst_op = self._decode_destination(address + 2 + src_op.len, Ad, dst, size)
            return {
                0x4: MovInst,
                0x5: AddInst,
                0x8: SubInst,
                0x9: CmpInst,
                0xa: DaddInst,
                0xd: BisInst,
                0xf: AndInst
            }[opc](src_op, dst_op, size)
            
    def get_pc(self):
        return self.registers[0]
    
    def set_pc(self, val):
        self.registers[0] = val & 0xffff
        
    def get_sr(self):
        return self.registers[2]
        
    def get_flag(self, flag):
        return self.get_sr() & (1 << flag.value) != 0
        
    def set_flag(self, flag, state):
        self.registers[2] = (self.registers[2] & ~(1 << flag.value)) | ((1 << flag.value) if state else 0)
        self.registers[2] &= 0x01ff
        
    def get_sp(self):
        return self.registers[1]
        
    def set_sp(self, val):
        self.registers[1] = val & 0xffff
        
    def get_register(self, idx):
        return self.registers[idx]
        
    def set_register(self, idx, val):
        self.registers[idx] = val
        
    def __str__(self):
        return ("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
             "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
             "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
             "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
             % (self.get_pc(), self.get_sp(), self.get_sr(), 
                self.get_register(3), self.get_register(4),
                self.get_register(5), self.get_register(6),
                self.get_register(7), self.get_register(8),
                self.get_register(9), self.get_register(10),
                self.get_register(11), self.get_register(12),
                self.get_register(13), self.get_register(14),
                self.get_register(15))
        
        
def parse_srec_line(line):
    type = line[:2]
    count = int(line[2:4], 16)
    address = int(line[4:8], 16)
    data = [int("".join(x), 16) for x in zip(*[iter(line[8:-2])] * 2)]
    checksum = int(line[-2:], 16)
    if (count + (address >> 8) + (address & 0xff) + sum(data) + checksum) & 0xff != 0xff:
        raise RuntimeError("Invalid SREC checksum: " + line)
    return (type, address, data)
        
def load_srec(filename, memory, cpu):
    with open(filename, 'r') as file:
        for line in file.readlines():
            data = parse_srec_line(line.strip())
            if data[0] == "S1":
                for i in range(len(data[2])):
                    memory.write(data[1] + i, 1, data[2][i])
            elif data[0] == "S9":
                cpu.set_pc(data[1])
            else:
                raise RuntimeError("Unknown SREC record type: " + data[0])
                
def restore_snapshot(filename, memory, cpu):
    with open(filename, 'r') as file:
        snapshot = json.load(file)
        snap_mem = snapshot["updatememory"]
        lines = map("".join, zip(*[iter(snap_mem)] * 36))
        for line in lines:
            addr = int(line[0:4], 16)
            data = [int("".join(x), 16) for x in zip(*[iter(line[4:])] * 2)]
            for (byte, i) in zip(data, range(1024)):
                memory.write(addr + i, 1, byte)
        cpu.registers = snapshot["regs"]
        

def main(args, env):
    memory = Memory()
    cpu = MSP430Cpu(memory)
    
    load_srec(args.image, memory, cpu)
    
    if args.restore:
        restore_snapshot(args.restore, memory, cpu)
    
    if args.verify:
        with open(args.verify, "r") as file:
            linenum = 1
            lastinst = None
            for line in file.readlines():
                data = json.loads(line.strip())
                #check that registers are correct
                if cpu.registers != data["regs"]:
                    print("========================")
                    print("Last instruction: %s" % lastinst)
                    print("CPU registers:")
                    print(str(cpu))
                    print("Trace registers:")
                    print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
                         "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
                         "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
                         "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
                         % (data["regs"][0], data["regs"][1], data["regs"][2], 
                            data["regs"][3], data["regs"][4], data["regs"][5],
                            data["regs"][6], data["regs"][7], data["regs"][8],
                            data["regs"][9], data["regs"][10], data["regs"][11],
                            data["regs"][12], data["regs"][13], data["regs"][14], 
                            data["regs"][15]))
                    raise RuntimeError("Mismatch between registers in tracefile line %d" % (linenum, ))
                for memline in ["".join(x) for x in zip(*[iter(data["updatememory"])] * 36)]:
                    addr = int(memline[0:4], 16)
                    bytes = [int("".join(x), 16) for x in zip(*[iter(memline[4:])] * 2)]
                    for i in range(len(bytes)):
                        if bytes[i] != memory.read(addr + i, 1):
                            print("========================")
                            print("Last instruction: %s" % lastinst)
                            print("CPU registers:")
                            print(str(cpu))
                            print("Trace registers:")
                            print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
                                 "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
                                 "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
                                 "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
                                 % (data["regs"][0], data["regs"][1], data["regs"][2], 
                                    data["regs"][3], data["regs"][4], data["regs"][5],
                                    data["regs"][6], data["regs"][7], data["regs"][8],
                                    data["regs"][9], data["regs"][10], data["regs"][11],
                                    data["regs"][12], data["regs"][13], data["regs"][14], 
                                    data["regs"][15]))
                            print("Memory in emulator:")
                            print("%04x: %s" % (addr, " ".join(["%02x" % memory.read(x, 1) for x in range(addr, addr + 16)])))
                            print("Memory in trace:")
                            print("%04x: %s" % (addr, " ".join(["%02x" % x for x in bytes])))
                            raise RuntimeError("Mismatch between memory contents at address 0x%04x in tracefile line %d" % (addr + i, linenum))
                print("CPU:")
                print(str(cpu))
                print("Trace:")
                print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
                     "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
                     "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
                     "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
                     % (data["regs"][0], data["regs"][1], data["regs"][2], 
                        data["regs"][3], data["regs"][4], data["regs"][5],
                        data["regs"][6], data["regs"][7], data["regs"][8],
                        data["regs"][9], data["regs"][10], data["regs"][11],
                        data["regs"][12], data["regs"][13], data["regs"][14], 
                        data["regs"][15]))
                cpu.step(True)
                lastinst = data["insn"]
                linenum += 1
    else:
        while True:
            print(str(cpu))
            cpu.step(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type = str, default = None, help = "Program image as SREC file")
    parser.add_argument("--verify", type = str, default = None, help = "Verify execution against given web trace")
    parser.add_argument("--restore", type = str, default = None, help = "Restore system snapshot from JSON file")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    code = main(parse_args(), os.environ)
    if code:
        sys.exit(code)
	
