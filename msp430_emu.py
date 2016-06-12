#!/usr/bin/env python3

import os
import sys
import argparse
import json

from enum import Enum

import llvmlite.ir as llir
import llvmlite.binding as llvm
import ctypes as cty

INT_BITS = 16
DATA_LAYOUT_STRING = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
TRIPLE = "x86_64-apple-darwin15.4.0"
    
class TranslationContext(object):
    def __init__(self):

        self.translation_blocks = {}
        self.function_type = llir.FunctionType(llir.VoidType(), (llir.PointerType(CpuStateStruct.llvm_type), ))
        self.target = llvm.Target.from_default_triple().create_target_machine() 
        self.counter = 0
    
class TranslationBlock(object):
    def __init__(self, context, address):
        self.context = context
        self.address = address
        self.instructions = []
        self.module = llir.Module("tb_0x%04x_%d" % (address, context.counter))
#        self.module.data_layout = DATA_LAYOUT_STRING
#        self.module.triple = TRIPLE
        self.function = llir.Function(self.module, context.function_type, "tb_0x%04x_%d" % (address, context.counter))
        context.counter += 1
        self.builder = llir.IRBuilder(self.function.append_basic_block("entry"))
        self.compiled_module = None
        self.native = None
        self.execution_engine = None
        self.target_machine = None
        context.translation_blocks[address] = self
        
    def get_flag(self, flag):
        gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_flag_gep_indices(flag), True, "ptr_flag_%s" % (flag.name, ))
        value = self.builder.load(gep, "flag_%s_i16" % (flag.name, ))
        return self.builder.trunc(value, llir.IntType(1), "flag_%s" % (flag.name, ))
        
    def set_flag(self, flag, value):
        gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_flag_gep_indices(flag), True, "ptr_flag_%s" % (flag.name, ))
        value = self.builder.zext(value, llir.IntType(INT_BITS), "flag_%s_i16" % (flag.name, ))
        self.builder.store(value, gep)
        
        #high bits are cleared
        gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_register_gep_indices(2), True, "ptr_r2")
        r2 = self.builder.load(gep, "r2")
        r2_no_callgate = self.builder.and_(r2, llir.Constant(llir.IntType(16), ~0xfe00), "r2_without_flags")
        self.builder.store(r2_no_callgate, gep)
        
    def get_register(self, reg):
        if reg == 2:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_register_gep_indices(2), True, "ptr_r2")
            r2 = self.builder.load(gep, "r2")
            r2 = self.builder.and_(r2, llir.Constant(llir.IntType(16), ~0x0107), "r2_without_flags")
            for flag in (CPUFlags.C, CPUFlags.Z, CPUFlags.N, CPUFlags.V):
                gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_flag_gep_indices(flag), True, "ptr_flag_%s" % (flag.name, ))
                flag_val = self.builder.load(gep, "flag_%s_i16" % (flag.name, ))
                flag_val = self.builder.shl(flag_val, llir.Constant(llir.IntType(16), flag.value), "flag_%s_shifted" % (flag.name, ))
                r2 = self.builder.or_(r2, flag_val, "r2")
            return r2
        else:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_register_gep_indices(reg), True, "ptr_r%d" % (reg, ))
            return self.builder.load(gep, "r%d" % (reg, ))
        
    def set_register(self, reg, value):
        if reg == 2:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_register_gep_indices(2), True, "ptr_r2")
            r2_no_flags = self.builder.and_(value, llir.Constant(llir.IntType(16), ~0x0107), "r2_without_flags")
            self.builder.store(r2_no_flags, gep)
            for flag in (CPUFlags.C, CPUFlags.Z, CPUFlags.N, CPUFlags.V):
                gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_flag_gep_indices(flag), True, "ptr_flag_%s" % (flag.name, ))
                flag_val = self.builder.lshr(value, llir.Constant(llir.IntType(INT_BITS), flag.value), "flag_%s_i16_unfiltered" % (flag.name, ))
                flag_val = self.builder.and_(flag_val, llir.Constant(llir.IntType(INT_BITS), 1), "flag_%s_i16" % (flag.name, ))
                self.builder.store(flag_val, gep)
        else:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_register_gep_indices(reg), True, "ptr_r%d" % (reg, ))
            if value.type.width == 8:
                value = self.builder.zext(value, llir.IntType(16))
            self.builder.store(value, gep)
            
    def read_memory(self, address):
        int32Ty = llir.IntType(32)
        int16Ty = llir.IntType(16)
        addr_val = self.builder.ptrtoint(address, int32Ty)
        if address.type.pointee.width == 8:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_val), True, "ptr_mem")
            return self.builder.load(gep)
        elif address.type.pointee.width == 16:
            gep_lo = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_val), True, "ptr_mem")
            addr_hi = self.builder.add(addr_val, llir.Constant(int32Ty, 1))
            gep_hi = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_hi), True, "ptr_mem")
            
            val_lo = self.builder.load(gep_lo)
            val_lo = self.builder.zext(val_lo, int16Ty)
            val_hi = self.builder.load(gep_hi)
            val_hi = self.builder.zext(val_hi, int16Ty)
            val_hi = self.builder.shl(val_hi, llir.Constant(int16Ty, 8))
            val = self.builder.or_(val_hi, val_lo)
            return val
        else:
            assert(False)
            
    def write_memory(self, address, value):
        int32Ty = llir.IntType(32)
        int16Ty = llir.IntType(16)
        addr_val = self.builder.ptrtoint(address, int32Ty)
        assert(address.type.pointee == value.type)
        if address.type.pointee.width == 8:
            gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_val), True, "ptr_mem")
            self.builder.store(value, gep)
        elif address.type.pointee.width == 16:
            val_lo = self.builder.trunc(value, llir.IntType(8))
            val_hi = self.builder.lshr(value, llir.Constant(value.type, 8))
            val_hi = self.builder.trunc(val_hi, llir.IntType(8))
            
            gep_lo = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_val), True, "ptr_mem")
            addr_hi = self.builder.add(addr_val, llir.Constant(addr_val.type, 1))
            gep_hi = self.builder.gep(self.function.args[0], CpuStateStruct.get_memory_gep_indices(addr_hi), True, "ptr_mem")
            
            self.builder.store(val_lo, gep_lo)
            self.builder.store(val_hi, gep_hi)
            
            #print_gep = self.builder.gep(self.function.args[0], [llir.Constant(llir.IntType(32), 0), llir.Constant(llir.IntType(32), 6)])
            #print_func = self.builder.load(print_gep, "print_func")
            #self.builder.call(print_func, [self.builder.trunc(addr_val, llir.IntType(16))])
            #self.builder.call(print_func, [self.builder.zext(val_lo, llir.IntType(16))])
        else:
            assert(False)
            
        #Invalidate instruction cache 
        invinst_gep = self.builder.gep(self.function.args[0], CpuStateStruct.get_invalidate_instruction_cache_gep_indices(), "ptr_invalidate_instruction_cache")
        invinst_func = self.builder.load(invinst_gep, "invalidate_instruction_cache")
        self.builder.call(invinst_func, [self.builder.ptrtoint(address, int16Ty)])
            
INVALIDATE_INSTRUCTION_CACHE_FUNC = cty.CFUNCTYPE(None, cty.c_uint16)
        
class CpuStateStruct(cty.Structure):
    
    fields = [
        ("registers", cty.c_uint16 * 16, llir.ArrayType(llir.IntType(INT_BITS), 16)),
        ("C", cty.c_uint16, llir.IntType(INT_BITS)),
        ("Z", cty.c_uint16, llir.IntType(INT_BITS)),
        ("N", cty.c_uint16, llir.IntType(INT_BITS)),
        ("V", cty.c_uint16, llir.IntType(INT_BITS)),
        ("memory", cty.c_uint8 * 0x10000, llir.ArrayType(llir.IntType(8), 0x10000)),
        ("invalidate_instruction_cache", INVALIDATE_INSTRUCTION_CACHE_FUNC, llir.PointerType(llir.FunctionType(llir.VoidType(), [llir.IntType(16)])))]
    
    _fields_ = [(name, ctype) for name, ctype, _ in fields]
        
#    llvm_type = llir.LiteralStructType([llvmtype for _, _, llvmtype in fields], True)
    llvm_type = llir.LiteralStructType([llvmtype for _, _, llvmtype in fields])
        
    @classmethod
    def get_flag_gep_indices(clazz, flag):
        flag_idx = {
            CPUFlags.C: 1,
            CPUFlags.Z: 2,
            CPUFlags.N: 3,
            CPUFlags.V: 4}[flag]
        return (llir.Constant(llir.IntType(32), 0), llir.Constant(llir.IntType(32), flag_idx))
            
    @classmethod    
    def get_register_gep_indices(clazz, reg):
        return (llir.Constant(llir.IntType(32), 0), 
                llir.Constant(llir.IntType(32), 0), 
                llir.Constant(llir.IntType(32), reg))
        
    @classmethod
    def get_memory_gep_indices(clazz, address):
        return (llir.Constant(llir.IntType(32), 0), 
                llir.Constant(llir.IntType(32), 5), 
                address)
    
    @classmethod
    def get_invalidate_instruction_cache_gep_indices(clazz):
        return (llir.Constant(llir.IntType(32), 0), 
                llir.Constant(llir.IntType(32), 6))
        
    def get_register(self, reg):
        if reg == 2:
            return (self.registers[2] & ~0x0107) | \
                (self.C << CPUFlags.C.value) | \
                (self.Z << CPUFlags.Z.value) | \
                (self.N << CPUFlags.N.value) | \
                (self.V << CPUFlags.V.value)
        else:
            return self.registers[reg]
            
    def set_register(self, reg, val):
        if reg == 2:
            self.C = (val >> CPUFlags.C.value) & 1
            self.Z = (val >> CPUFlags.Z.value) & 1
            self.N = (val >> CPUFlags.N.value) & 1
            self.V = (val >> CPUFlags.V.value) & 1
            self.registers[2] = val & ~0x0107
        else:
            self.registers[reg] = val
            
    def get_flag(self, flag):
        return {
            CPUFlags.C: self.C,
            CPUFlags.Z: self.Z,
            CPUFlags.N: self.N,
            CPUFlags.V: self.V}[flag]
            
    def set_flag(self, flag, val):
        setattr(self, flag.name, val)
        
    def read_memory(self, address, size):
        if size == 1:
            return self.memory[address]
        elif size == 2:
            return self.memory[address] | (self.memory[address + 1] << 8)
    
    def write_memory(self, address, size, value):
        if size == 1:
            self.memory[address] = value
        elif size == 2:
            self.memory[address] = value & 0xff
            self.memory[address + 1] = (value >> 8) & 0xff
            
    def is_cpuoff(self):
        return self.registers[2] & (1 << CPUFlags.CPUOFF.value) != 0

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
            
    def _advance_pc(self, tb):
        pc = tb.get_register(0)
        pc = tb.builder.add(pc, llir.Constant(pc.type, self.len))
        tb.set_register(0, pc)

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
            
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        
        # Get the jump condition
        cond = {
            Condition.not_equal: lambda: tb.builder.not_(tb.get_flag(CPUFlags.Z)),
            Condition.equal: lambda: tb.get_flag(CPUFlags.Z),
            Condition.no_carry: lambda: tb.builder.not_(tb.get_flag(CPUFlags.C)),
            Condition.carry: lambda: tb.get_flag(CPUFlags.C),
            Condition.negative: lambda: tb.get_flag(CPUFlags.N),
            Condition.greater_equal: lambda: tb.builder.icmp_unsigned('==', tb.get_flag(CPUFlags.N), tb.get_flag(CPUFlags.V)),
            Condition.less: lambda: tb.builder.icmp_unsigned('!=', tb.get_flag(CPUFlags.N), tb.get_flag(CPUFlags.V)),
            Condition.always: lambda: llir.Constant(llir.IntType(1), 1)}[self.condition]()
        
        bb_true = tb.function.append_basic_block("cond_true")
        bb_false = tb.function.append_basic_block("cond_false")
        
        tb.builder.cbranch(cond, bb_true, bb_false)
        
        builder_false = llir.IRBuilder(bb_false)
        builder_false.ret_void()
        
        tb.builder = llir.IRBuilder(bb_true)
        pc = tb.get_register(0)
        new_pc = tb.builder.add(pc, llir.Constant(pc.type, self.offset))
        tb.set_register(0, new_pc)
        tb.builder.ret_void()
        
        return False
        
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
            
        offset = self.offset + 2
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
        
    def set_llvm_value(self, tb, value):
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
        if self.reg == 0:
            return (cpu.get_register(self.reg) + 2) & mask
        else:
            return cpu.get_register(self.reg) & mask
            
    def get_llvm_value(self, tb):
        reg = tb.get_register(self.reg)
        if self.size == 1:
            reg = tb.builder.trunc(reg, llir.IntType(8))
        return reg
        
    def set(self, cpu, val):
        mask = 0xff if self.size == 1 else 0xffff
        cpu.set_register(self.reg, val & mask)
        
    def set_llvm_value(self, tb, val):
        if val.type.width == 8:
            val = tb.builder.zext(val, llir.IntType(INT_BITS))
        tb.set_register(self.reg, val)
        
    def __str__(self):
        return self._regname(self.reg)
            
class AbsoluteOperand(Operand):
    def __init__(self, address, size):
        self.address = address
        self.size = size
        self.len = 2
        
    def get(self, cpu):
        return cpu.memory.read(self.address, self.size)
        
    def get_llvm_value(self, tb):
        addr = llir.Constant(llir.IntType(INT_BITS), self.address)
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        return tb.read_memory(addr)
        
    def set(self, cpu, val):
        cpu.memory.write(self.address, self.size, val)
        
    def set_llvm_value(self, tb, value):
        addr = llir.Constant(llir.IntType(INT_BITS), self.address)
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        tb.write_memory(addr, value)
        
    def __str__(self):
        return "&%04x" % self.address
        
class ConstantOperand(Operand):
    def __init__(self, val, size):
        self.val = val
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        return self.val
        
    def get_llvm_value(self, tb):
        return llir.Constant(llir.IntType(self.size * 8), self.val)
        
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
        
    def get_llvm_value(self, tb):
        reg = tb.get_register(self.reg)
        addr = tb.builder.add(reg, llir.Constant(reg.type, self.offset))
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        return tb.read_memory(addr)
        
    def set_llvm_value(self, tb, val):
        reg = tb.get_register(self.reg)
        addr = tb.builder.add(reg, llir.Constant(reg.type, self.offset))
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        tb.write_memory(addr, val)
        
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
        
    def get_llvm_value(self, tb):
        reg = tb.get_register(self.reg)
        addr = tb.builder.inttoptr(reg, llir.PointerType(llir.IntType(self.size * 8)))
        return tb.read_memory(addr)
        
    def __str__(self):
        return "@%s" % (self._regname(self.reg), )
        
class PcIndirectOperand(Operand):
    def __init__(self, size):
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        return cpu.memory.read(cpu.get_register(0) + 2, self.size)
        
    def get_llvm_value(self, tb):
        addr = tb.get_register(0)
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        return tb.read_memory(addr)
        
    def __str__(self):
        return "@%s" % (self._regname(0), )
        
class RegisterIndirectIncrementOperand(Operand):
    def __init__(self, reg, size):
        self.reg = reg
        self.size = size
        self.len = 0
        
    def get(self, cpu):
        value = cpu.memory.read(cpu.get_register(self.reg), self.size)
        cpu.set_register(self.reg, cpu.get_register(self.reg) + self.size)
        return value
        
    def get_llvm_value(self, tb):
        addr = tb.get_register(self.reg)
        tb.set_register(self.reg, tb.builder.add(addr, llir.Constant(addr.type, self.size)))
        addr = tb.builder.inttoptr(addr, llir.PointerType(llir.IntType(self.size * 8)))
        return tb.read_memory(addr)
        
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
        
    def get_llvm_value(self, tb):
        return llir.Constant(llir.IntType(self.size * 8), self.val)
        
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
        val = self.op.get(cpu)
        msb = val & (0x80 if self.size == 1 else 0x8000)
        result = (val >> 1) | msb
#        cpu.set_flag(CPUFlags.C, val & 1 != 0)
        cpu.set_flag(CPUFlags.N, cpu.get_flag(CPUFlags.N) or msb != 0)
        cpu.set_flag(CPUFlags.Z, 0)
        cpu.set_flag(CPUFlags.V, False)
        self.op.set(cpu, result)
        super().eval(cpu)
        
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        c = tb.get_flag(CPUFlags.C)
        old_c = tb.builder.zext(c, llir.IntType(self.size * 8))
        old_c = tb.builder.shl(old_c, llir.Constant(old_c.type, self.size * 8 - 1))
        val = self.op.get_llvm_value(tb)
        msb = tb.builder.and_(val, llir.Constant(val.type, 1 << (val.type.width - 1)))
        result = tb.builder.lshr(val, llir.Constant(val.type, 1))
        result = tb.builder.or_(result, msb)
        self.op.set_llvm_value(tb, result)
        tb.set_flag(CPUFlags.V, llir.Constant(llir.IntType(1), 0))
        tb.set_flag(CPUFlags.N, tb.builder.or_(tb.get_flag(CPUFlags.N), tb.builder.icmp_unsigned("!=", msb, llir.Constant(msb.type, 0))))
        tb.set_flag(CPUFlags.Z, llir.Constant(llir.IntType(1), 0))
        
        
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
        cpu.set_flag(CPUFlags.N, cpu.get_flag(CPUFlags.N) or old_c != 0)
        cpu.set_flag(CPUFlags.Z, cpu.get_flag(CPUFlags.Z) and result == 0)
        cpu.set_flag(CPUFlags.V, False)
        self.op.set(cpu, result)
        super().eval(cpu)
        
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        c = tb.get_flag(CPUFlags.C)
        old_c = tb.builder.zext(c, llir.IntType(self.size * 8))
        old_c = tb.builder.shl(old_c, llir.Constant(old_c.type, self.size * 8 - 1))
        val = self.op.get_llvm_value(tb)
        result = tb.builder.lshr(val, llir.Constant(val.type, 1))
        result = tb.builder.or_(result, old_c)
        self.op.set_llvm_value(tb, result)
        tb.set_flag(CPUFlags.C, tb.builder.trunc(val, llir.IntType(1)))
        tb.set_flag(CPUFlags.V, llir.Constant(llir.IntType(1), 0))
        tb.set_flag(CPUFlags.N, tb.builder.or_(tb.get_flag(CPUFlags.N), c))
        tb.set_flag(CPUFlags.Z, tb.builder.and_(tb.get_flag(CPUFlags.Z), tb.builder.icmp_unsigned("==", result, llir.Constant(result.type, 0))))
        
class SXTInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "sxt"
        super().__init__(op, size)
        
    def eval(self, cpu):
        val = self.op.get(cpu)
        result = val | (0xff00 if val & 0x80 != 0 else 0)
        cpu.set_flag(CPUFlags.V, False)
        cpu.set_flag(CPUFlags.C, result != 0)
        cpu.set_flag(CPUFlags.N, result & 0x8000 != 0)
        cpu.set_flag(CPUFlags.Z, result == 0)
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
        
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        sp = tb.get_register(1)
        sp = tb.builder.sub(sp, llir.Constant(sp.type, 2))
        tb.set_register(1, sp)
        tb.write_memory(tb.builder.inttoptr(sp, llir.PointerType(llir.IntType(16))), self.op.get_llvm_value(tb))
        
        
class CallInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "call"
        super().__init__(op, size)
        
    def eval(self, cpu):
        cpu.set_sp(cpu.get_sp() - 2)
        cpu.memory.write(cpu.get_sp(), self.size, cpu.get_pc() + self.len)
        cpu.set_pc(self.op.get(cpu))
        
    def generate_llvm(self, tb):
        sp = tb.get_register(1)
        sp = tb.builder.sub(sp, llir.Constant(sp.type, 2))
        tb.set_register(1, sp)
        address = tb.builder.inttoptr(sp, llir.PointerType(llir.IntType(16)))
        pc = tb.get_register(0)
        next_pc = tb.builder.add(pc, llir.Constant(pc.type, self.len))
        tb.write_memory(address, next_pc)
        tb.set_register(0, self.op.get_llvm_value(tb))
        
class SwpbInst(UnaryInst):
    def __init__(self, op, size):
        self.mnem = "swpb"
        super().__init__(op, size)
        
    def eval(self, cpu):
        val = self.op.get(cpu)
        self.op.set(cpu, (val >> 8) | ((val & 0xff) << 8))
        super().eval(cpu)
        
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        val = self.op.get_llvm_value(tb)
        hi = tb.builder.lshr(val, llir.Constant(val.type, 8))
        lo = tb.builder.and_(val, llir.Constant(val.type, 0xff))
        new_val = tb.builder.shl(lo, llir.Constant(lo.type, 8))
        new_val = tb.builder.or_(new_val, hi)
        self.op.set_llvm_value(tb, new_val)
        
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
        if not (isinstance(self.destination, RegisterOperand) and self.destination.reg == 0 and result != None):
            #Only advance program counter if it hasn't been modified by the instruction
            super().eval(cpu)
            
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        src = lambda: self.source.get_llvm_value(tb)
        dst = lambda: self.destination.get_llvm_value(tb)
        
        result = self._generate_llvm(tb, src, dst)
        
        if not result is None:
            self.destination.set_llvm_value(tb, result)
            
            
    def _set_logical_flags(self, tb, flags, result):
        msb = tb.builder.and_(result, llir.Constant(result.type, 1 << (result.type.width - 1)))
        zero = llir.Constant(result.type, 0)
        if "V" in flags:
            tb.set_flag(CPUFlags.V, llir.Constant(llir.IntType(1), 0))
        if "C" in flags:
            tb.set_flag(CPUFlags.C, tb.builder.icmp_unsigned('!=', result, zero))
        if "Z" in flags:
            tb.set_flag(CPUFlags.Z, tb.builder.icmp_unsigned('==', result, zero))
        if "N" in flags:
            tb.set_flag(CPUFlags.N, tb.builder.icmp_unsigned('!=', msb, zero))
            
    def _set_arithmetic_flags(self, tb, flags, result, dst, carry):
        if 'C' in flags:
            tb.set_flag(CPUFlags.C, carry)
        if 'V' in flags:
            tb.set_flag(CPUFlags.V, llir.Constant(llir.IntType(1), 0))
#            dst_N = tb.builder.icmp_unsigned("!=", 
#                tb.builder.and_(dst, llir.Constant(dst.type, 1 << (dst.type.width - 1))),
#                llir.Constant(dst.type, 0))
#            res_N = tb.builder.icmp_unsigned("!=", 
#                tb.builder.and_(result, llir.Constant(result.type, 1 << (result.type.width - 1))),
#                llir.Constant(dst.type, 0))
#            tb.set_flag(CPUFlags.V, tb.builder.xor(dst_N, res_N))
        if 'N' in flags:
            res_N = tb.builder.icmp_unsigned("!=", 
                tb.builder.and_(result, llir.Constant(result.type, 1 << (result.type.width - 1))),
                llir.Constant(result.type, 0))
            tb.set_flag(CPUFlags.N, res_N)
        if 'Z' in flags:
            tb.set_flag(CPUFlags.Z, tb.builder.icmp_unsigned("==", result, llir.Constant(result.type, 0)))
        
    def __str__(self):
        mnem = self.mnem + (".b" if self.size == 1 else "")
        return "%s %s, %s" % (mnem, str(self.source), str(self.destination))
        
class MovInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "mov"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src(), "")
        
    def _generate_llvm(self, tb, src, dst):
        return src()
        
        
class AndInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "and"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        result = src() & dst()
        cpu.set_flag(CPUFlags.V, False)
        cpu.set_flag(CPUFlags.C, result != 0)
        return (result, "NZ")
        
    def _generate_llvm(self, tb, src, dst):
        result = tb.builder.and_(src(), dst())
        self._set_logical_flags(tb, "CVNZ", result)
        return result
        
        
class XorInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "xor"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        result = src() ^ dst()
        msb = 0x80 if self.size == 1 else 0x8000
        cpu.set_flag(CPUFlags.V, False)
        cpu.set_flag(CPUFlags.C, result != 0)
        return (result, "NZ")
    
    def _generate_llvm(self, tb, src, dst):
        result = tb.builder.xor(src(), dst())
        self._set_logical_flags(tb, "CVNZ", result)
        return result
        
class BisInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "bis"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src() | dst(), "")
        
    def _generate_llvm(self, tb, src, dst):
        return tb.builder.or_(src(), dst())

        
class BicInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "bic"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (~src() & dst(), "")
        
class CmpInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "cmp"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        result = (~src() & (0xff if self.size == 1 else 0xffff)) + 1 + dst()
        self._set_flags(cpu, "CVNZ", result, dst())
        return (None, "")
        
    def _generate_llvm(self, tb, src, dst):
        s = src()
        d = dst()
        
        assert(s.type == d.type)
        
        a = tb.builder.zext(tb.builder.not_(s), llir.IntType(s.type.width + 1))
        b = tb.builder.zext(d, llir.IntType(s.type.width + 1))
        
        result_carry = tb.builder.add(
            tb.builder.add(
                a,
                b),
            llir.Constant(a.type, 1))
            
        result = tb.builder.trunc(result_carry, s.type)
        carry = tb.builder.lshr(result_carry, llir.Constant(result_carry.type, s.type.width))
        carry = tb.builder.trunc(carry, llir.IntType(1))
        self._set_arithmetic_flags(tb, "CVNZ", result, d, carry)
        return None
        
class SubInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "sub"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return ((~src() & (0xff if self.size == 1 else 0xffff)) + 1 + dst(), "CVNZ")
        
    def _generate_llvm(self, tb, src, dst):
        s = src()
        d = dst()
        
        assert(s.type == d.type)
        
        a = tb.builder.zext(tb.builder.not_(s), llir.IntType(s.type.width + 1))
        b = tb.builder.zext(d, llir.IntType(s.type.width + 1))
        
        result_carry = tb.builder.add(
            tb.builder.add(
                a,
                b),
            llir.Constant(a.type, 1))
            
        result = tb.builder.trunc(result_carry, s.type)
        carry = tb.builder.lshr(result_carry, llir.Constant(result_carry.type, s.type.width))
        carry = tb.builder.trunc(carry, llir.IntType(1))
        self._set_arithmetic_flags(tb, "CVNZ", result, d, carry)
        return result

class AddInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "add"
        super().__init__(source, dest, size)
        
    def _eval(self, cpu, src, dst):
        return (src() + dst(), "CVNZ")
        
    def _generate_llvm(self, tb, src, dst):
        s = src()
        d = dst()
        
        assert(s.type == d.type)
        
        a = tb.builder.zext(s, llir.IntType(s.type.width + 1))
        b = tb.builder.zext(d, llir.IntType(s.type.width + 1))
        
        result_carry = tb.builder.add(a, b)
            
        result = tb.builder.trunc(result_carry, s.type)
        carry = tb.builder.lshr(result_carry, llir.Constant(result_carry.type, s.type.width))
        carry = tb.builder.trunc(carry, llir.IntType(1))
        self._set_arithmetic_flags(tb, "CVNZ", result, d, carry)
        return result
        
class DaddInst(BinaryInst):
    def __init__(self, source, dest, size):
        self.mnem = "dadd"
        super().__init__(source, dest, size)
        
    def eval(self, cpu):
        src = self.source.get(cpu)
        dst = self.destination.get(cpu)
        msb = 0x80 if self.size == 1 else 0x8000
        max_val = 99 if self.size == 1 else 9999
        
        result = 0
        n = 0
        carry = 0 #1 if cpu.get_flag(CPUFlags.C) else 0
        for digit in range(self.size * 2):
            a = (src >> (digit * 4)) & 0xf
            b = (dst >> (digit * 4)) & 0xf
            
            tmp = a + b + carry
            n = tmp & 0x8 != 0
            if tmp > 9:
                tmp -= 10
                carry = 1
            else:
                carry = 0
            result = result | ((tmp & 0xf) << (digit * 4))

        self.destination.set(cpu, result)
        cpu.set_flag(CPUFlags.N, cpu.get_flag(CPUFlags.N) or (result & msb != 0))
        #cpu.set_flag(CPUFlags.C, carry != 0)
        cpu.set_flag(CPUFlags.C, carry != 0)
        cpu.set_flag(CPUFlags.Z, result == 0)
        cpu.set_flag(CPUFlags.V, 0)
        MSP430Instruction.eval(self, cpu)
        
    def generate_llvm(self, tb):
        self._advance_pc(tb)
        src = self.source.get_llvm_value(tb)
        dst = self.destination.get_llvm_value(tb)
        typ = llir.IntType(self.size * 8)
        bitpos = llir.Constant(typ, 0)
        carry = llir.Constant(llir.IntType(1), 0)
        n = llir.Constant(llir.IntType(1), 0)
        result = llir.Constant(src.type, 0)
        
        bb_cond = tb.function.append_basic_block("dadd_condition")
        tb.builder.branch(bb_cond)
        bb_body = tb.function.append_basic_block("dadd_loop_body")
        bb_after = tb.function.append_basic_block("dadd_after_loop")
        builder_cond = llir.IRBuilder(bb_cond)
        bitpos_phi = builder_cond.phi(bitpos.type, "bitpos")
        bitpos_phi.add_incoming(bitpos, tb.builder.block)
        carry_phi = builder_cond.phi(carry.type, "carry")
        carry_phi.add_incoming(carry, tb.builder.block)
        n_phi = builder_cond.phi(n.type, "n")
        n_phi.add_incoming(n, tb.builder.block)
        result_phi = builder_cond.phi(result.type, "result")
        result_phi.add_incoming(result, tb.builder.block)
        exit_cond = builder_cond.icmp_unsigned(">=", bitpos_phi, llir.Constant(bitpos_phi.type, self.size * 8), "exit_cond")
        builder_cond.cbranch(exit_cond, bb_after, bb_body)
        
        builder_body = llir.IRBuilder(bb_body)
        a = builder_body.and_(builder_body.lshr(src, bitpos_phi), llir.Constant(src.type, 0xf))
        b = builder_body.and_(builder_body.lshr(dst, bitpos_phi), llir.Constant(src.type, 0xf))
        
        tmp = builder_body.add(a, b)
        tmp = builder_body.add(tmp, builder_body.zext(carry_phi, tmp.type))
        n = builder_body.icmp_unsigned("!=", builder_body.and_(tmp, llir.Constant(tmp.type, 0x8)), llir.Constant(tmp.type, 0))
        v = builder_body.icmp_unsigned(">", tmp, llir.Constant(tmp.type, 9))
        bb_overflow = tb.function.append_basic_block("overflow")
        bb_body2 = tb.function.append_basic_block("dadd_loop_body_2")
        carry_1 = llir.Constant(llir.IntType(1), 0)
        builder_body.cbranch(v, bb_overflow, bb_body2)
        builder_v = llir.IRBuilder(bb_overflow)
        tmp_2 = builder_v.sub(tmp, llir.Constant(tmp.type, 10))
        carry_2 = llir.Constant(llir.IntType(1), 1)
        builder_v.branch(bb_body2)
        
        builder_body2 = llir.IRBuilder(bb_body2)
        carry_phi2 = builder_body2.phi(carry_1.type, "carry")
        carry_phi2.add_incoming(carry_1, bb_body)
        carry_phi2.add_incoming(carry_2, bb_overflow)
        carry_phi.add_incoming(carry_phi2, bb_body2)
        n_phi.add_incoming(n, bb_body2)
        tmp_phi = builder_body2.phi(tmp.type, "tmp")
        tmp_phi.add_incoming(tmp, bb_body)
        tmp_phi.add_incoming(tmp_2, bb_overflow)
        tmp_phi = builder_body2.and_(tmp_phi, llir.Constant(tmp_phi.type, 0xf))
        tmp_phi = builder_body2.shl(tmp_phi, bitpos_phi)
        result = builder_body2.or_(result_phi, tmp_phi)
        result_phi.add_incoming(result, bb_body2)
        bitpos = builder_body2.add(bitpos_phi, llir.Constant(bitpos_phi.type, 4))
        bitpos_phi.add_incoming(bitpos, bb_body2)
        builder_body2.branch(bb_cond)
        
        tb.builder = llir.IRBuilder(bb_after)
        self.destination.set_llvm_value(tb, result_phi)
        n = tb.builder.or_(tb.get_flag(CPUFlags.N), tb.builder.trunc(tb.builder.lshr(result, llir.Constant(result.type, result.type.width - 1)), llir.IntType(1)))
        tb.set_flag(CPUFlags.N, n)
        tb.set_flag(CPUFlags.C, carry_phi)
        tb.set_flag(CPUFlags.Z, tb.builder.icmp_unsigned("==", result, llir.Constant(result.type, 0)))
        tb.set_flag(CPUFlags.V, llir.Constant(llir.IntType(1), 0))
        tb.builder.ret_void()

class Memory():
    def __init__(self):
        self.data = [0] * 0x10000
        
    def read(self, address, size):
        return {
            1: self.data[address & 0xffff],
            2: self.data[address & 0xffff] | (self.data[(address + 1) & 0xffff] << 8)
        }[size]

    def write(self, address, size, val):
        if address == 0x43fc:
            print("Address has been written to")
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
    def __init__(self, startaddr = 0x4400):
#        self.memory = memory
#        self.registers = [startaddr] + [0] * 15
#        self.memory.write(0x10, 2, 0x4130)
        self.state = CpuStateStruct()
        self.state.write_memory(0x10, 2, 0x4130)
        self.state.set_register(0, 0x4400)
        self.state.invalidate_instruction_cache = INVALIDATE_INSTRUCTION_CACHE_FUNC(self._invalidate_instruction_cache)
        self.translation_context = TranslationContext()
        self.translation_buffer = {}
        
    def step(self, verbose = False):
        address = self.state.get_register(0)
        if address == 0x0010 and self.state.get_register(2) & 0x8000 != 0:
            self._handle_callgate()
        else:
            if address in self.translation_buffer:
                tb = self.translation_buffer[address]
            else:
                tb = self.translate_block(address, single_instruction = True)
        #        print(str(self.translation_context.module))
   
                #print(str(tb.module))
                tb.compiled_module = llvm.parse_assembly(str(tb.module))
                tb.target_machine = llvm.Target.from_default_triple().create_target_machine()
                tb.execution_engine = llvm.create_mcjit_compiler(tb.compiled_module, tb.target_machine)
                tb.execution_engine.finalize_object()
                tb.native = cty.CFUNCTYPE(None, cty.POINTER(CpuStateStruct))(tb.execution_engine.get_pointer_to_function(tb.compiled_module.get_function(tb.function.name)))
                self.translation_buffer[address] = tb
        #            print("Before:")
        #            print(str(self))
            assert(tb.native is not None)
            if verbose:
                print(tb.instructions[0])
                #print(str(self))
#                print("%04x: %s" % (0xdff0, " ".join(["%02x%02x" % (x[1], x[0]) for x in zip(*[iter(self.state.memory[0xdff0:0xdff0+16])] * 2)])))
#                print("%04x: %s" % (0xe000, " ".join(["%02x%02x" % (x[1], x[0]) for x in zip(*[iter(self.state.memory[0xe000:0xe000+16])] * 2)])))
            tb.native(self.state)
#            print("After:")
#            print(str(self))
        
        # if self.get_pc() == 0x10 and self.get_sr() & 0x8000 != 0: #This is the call gate
        #     self._handle_callgate()
        # elif self.get_sr() & (1 << CPUFlags.CPUOFF.value) != 0:
        #     print("CPUOFF bit set. Terminating.")
        #     sys.exit(0)
        # else:
        #     inst = self.decode_instruction(self.get_pc())
        #     if verbose:
        #         print("%04x: %s" % (self.get_pc(), str(inst)))
        #     inst.eval(self)
            
    def translate_block(self, address, single_instruction = False):
        address = self.state.get_register(0)
#        assert(not (address in self.translation_context.translation_blocks))
        
        tb = TranslationBlock(self.translation_context, address)
        
        translate_next_instruction = True
        while translate_next_instruction:
            inst = self.decode_instruction(address)
            tb.instructions.append(inst)
            translate_next_instruction = inst.generate_llvm(tb)
            if single_instruction and translate_next_instruction:
                break
        if not tb.builder.basic_block.is_terminated:
            tb.builder.ret_void()
        return tb
        
        
        
            
    def _handle_callgate(self):
        gate_num = (self.state.get_register(2) >> 8) & 0x7f
        print("Invoking callgate 0x%x" % gate_num)
        if gate_num == 0:
            char = self.state.read_memory(self.state.get_register(1) + 8, 2)
            sys.stdout.write(chr(char))
        elif gate_num == 1:
            self.state.set_register(15, ord(sys.stdin.read(1)))
        elif gate_num == 2:
            buf_addr = self.state.read_memory(self.state.get_register(1) + 8, 2)
            buf_size = self.state.read_memory(self.state.get_register(1) + 10, 2)
            
            print("Writing input '%s' to 0x%x (max len 0x%x)" % ("".join([chr(x) for x in self.input]), buf_addr, buf_size))
            
            for (byte, i) in zip(self.input, range(buf_size)):
                self.state.write_memory(buf_addr + i, 1, byte)
        elif gate_num == 0x20:
            print("Generating random number")
            self.state.set_register(15, 0)
        elif gate_num == 0x7f:
            sys.stderr.write("Deadbolt unlocked. You're done.")
            sys.exit(0)
        else:
            print("Unhandled callgate code: 0x%02x" % gate_num)
            assert(False)
        
        self.state.set_register(0, self.state.read_memory(self.state.get_register(1), 2))
        self.state.set_register(1, self.state.get_register(1) + 2)
        #self.state.set_register(2, self.state.get_register(2) & 0xff)
                
        
    def _decode_source(self, address, As, reg, size):
        if reg == 0:
            return {
                0: lambda: RegisterOperand(reg, size),
                1: lambda: RegisterOffsetOperand(reg, self.state.read_memory(address, 2), size),
                2: lambda: PcIndirectOperand(size),
                3: lambda: ImmediateOperand(self.state.read_memory(address, size), size)}[As]()
            assert(False)
        elif reg == 2:
            return {
                0: lambda: RegisterOperand(reg, size),
                1: lambda: AbsoluteOperand(self.state.read_memory(address, 2), size),
                2: lambda: ConstantOperand(4, size),
                3: lambda: ConstantOperand(8, size)}[As]()
        elif reg == 3:
            return {
                0: lambda: ConstantOperand(0, size),
                1: lambda: ConstantOperand(1, size),
                2: lambda: ConstantOperand(2, size),
                3: lambda: ConstantOperand(0xffff, size)}[As]()
            assert(False)
        else:
            return {
                0: lambda: RegisterOperand(reg, size),
                1: lambda: RegisterOffsetOperand(reg, self.state.read_memory(address, 2), size),
                2: lambda: RegisterIndirectOperand(reg, size),
                3: lambda: RegisterIndirectIncrementOperand(reg, size)}[As]()
    
    def _decode_destination(self, address, Ad, reg, size):
        if reg == 0:
            return {
                0: lambda: RegisterOperand(reg, size)
            }[Ad]()
        elif reg == 2:
            return {
                0: lambda: RegisterOperand(reg, size),
                1: lambda: AbsoluteOperand(self.state.read_memory(address, size), size)
            }[Ad]()
        elif reg == 3:
            return DummyOperand()
        else:
            return {
                0: lambda: RegisterOperand(reg, size),
                1: lambda: RegisterOffsetOperand(reg, self.state.read_memory(address, 2), size)}[Ad]()
        
    def decode_instruction(self, address):
        # see http://www.ece.utep.edu/courses/web3376/Links_files/MSP430%20Quick%20Reference.pdf
        # and http://www.ti.com/lit/ug/slau144j/slau144j.pdf
        opcode = self.state.read_memory(address, 2)
        if opcode & 0xe000 == 0x2000:
            return JumpInst((sext(opcode & 0x3ff, 10) * 2) & 0xffff, Condition((opcode >> 10) & 7))
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
                    0x3: SXTInst,
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
            try:
                return {
                    0x4: MovInst,
                    0x5: AddInst,
                    0x8: SubInst,
                    0x9: CmpInst,
                    0xa: DaddInst,
                    0xc: BicInst,
                    0xd: BisInst,
                    0xe: XorInst,
                    0xf: AndInst
                }[opc](src_op, dst_op, size)
            except KeyError:
                print("Unknown assembler instruction 0x%04x at address 0x%04x" % (opcode, address))
                sys.exit(1)
        
    def _parse_srec_line(self, line):
        type = line[:2]
        count = int(line[2:4], 16)
        address = int(line[4:8], 16)
        data = [int("".join(x), 16) for x in zip(*[iter(line[8:-2])] * 2)]
        checksum = int(line[-2:], 16)
        if (count + (address >> 8) + (address & 0xff) + sum(data) + checksum) & 0xff != 0xff:
            raise RuntimeError("Invalid SREC checksum: " + line)
        return (type, address, data)
        
    def load_srec(self, filename):
        with open(filename, 'r') as file:
            for line in file.readlines():
                data = self._parse_srec_line(line.strip())
                if data[0] == "S1":
                    for i in range(len(data[2])):
                        self.state.write_memory(data[1] + i, 1, data[2][i])
                elif data[0] == "S9":
                    self.state.set_register(0, data[1])
                else:
                    raise RuntimeError("Unknown SREC record type: " + data[0])
                    
    def _invalidate_instruction_cache(self, address):
        address = address & ~1
        if address in self.translation_buffer:
            del self.translation_buffer[address]
        
    def __str__(self):
        return ("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
             "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
             "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
             "r12 %04x    r13 %04x    r14 %04x    r15 %04x    flags %s") \
             % tuple([self.state.registers[x] for x in range(16)] + \
               ["".join([x.name for x in (CPUFlags.C, CPUFlags.V, CPUFlags.N, CPUFlags.Z) if self.state.get_flag(x)])])
                
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
        
def registers_equal(x, y):
    return all([x.state.registers[i] == y[i] for i in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)]) and \
        ((x.state.registers[2] & ~0x0107) == (y[2] & ~0x0107)) and \
        (y[2] & (1 << CPUFlags.C.value) != 0) == (x.state.C != 0) and \
        (y[2] & (1 << CPUFlags.Z.value) != 0) == (x.state.Z != 0) and \
        (y[2] & (1 << CPUFlags.N.value) != 0) == (x.state.N != 0) and \
        (y[2] & (1 << CPUFlags.V.value) != 0) == (x.state.V != 0)

cpu = None        

def print_val(val):
    print("print_val: 0x%04x, mem[val] = 0x%04x" % (val, cpu.state.read_memory(val, 2)))
    #print("%04x: %s" % (0xdff0, " ".join(["%02x%02x" % (x[1], x[0]) for x in zip(*[iter(cpu.state.memory[0xdff0:0xdff0+16])] * 2)])))
    #print("%04x: %s" % (0xe000, " ".join(["%02x%02x" % (x[1], x[0]) for x in zip(*[iter(cpu.state.memory[0xe000:0xe000+16])] * 2)])))

def main(args, env):
    global cpu
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    cpu = MSP430Cpu()

    if args.hex:
        cpu.input = [int("".join(x), 16) for x in zip(*[iter(args.input)] * 2)]
    else:
        cpu.input = [ord(x) for x in args.input]
        
    cpu.load_srec(args.image)
    
    if args.restore:
        restore_snapshot(args.restore, memory, cpu)
    
    if args.verify:
        with open(args.verify, "r") as file:
            linenum = 1
            lastinst = None
            for line in file.readlines():
                data = json.loads(line.strip())
                #check that registers are correct
                if not registers_equal(cpu, data["regs"]):
                    print("========================")
                    print("Last instruction: %s" % lastinst)
                    print("CPU registers:")
                    print(str(cpu))
                    print("Trace registers:")
                    print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
                         "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
                         "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
                         "r12 %04x    r13 %04x    r14 %04x    r15 %04x    flags %s\n") \
                         % (data["regs"][0], data["regs"][1], data["regs"][2] & ~0x107, 
                            data["regs"][3], data["regs"][4], data["regs"][5],
                            data["regs"][6], data["regs"][7], data["regs"][8],
                            data["regs"][9], data["regs"][10], data["regs"][11],
                            data["regs"][12], data["regs"][13], data["regs"][14], 
                            data["regs"][15], "".join(x.name for x in (CPUFlags.C, CPUFlags.Z, CPUFlags.N, CPUFlags.V) if (data["regs"][2] & (1 << x.value)) != 0)))
                    raise RuntimeError("Mismatch between registers in tracefile line %d" % (linenum, ))
                # for memline in ["".join(x) for x in zip(*[iter(data["updatememory"])] * 36)]:
                #     addr = int(memline[0:4], 16)
                #     bytes = [int("".join(x), 16) for x in zip(*[iter(memline[4:])] * 2)]
                #     for i in range(len(bytes)):
                #          if bytes[i] != cpu.state.read_memory(addr + i, 1):
                #              print("========================")
                #              print("Last instruction: %s" % lastinst)
                #              print("CPU registers:")
                #              print(str(cpu))
                #              print("Trace registers:")
                #              print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
                #                   "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
                #                   "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
                #                   "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
                #                   % (data["regs"][0], data["regs"][1], data["regs"][2],
                #                      data["regs"][3], data["regs"][4], data["regs"][5],
                #                      data["regs"][6], data["regs"][7], data["regs"][8],
                #                      data["regs"][9], data["regs"][10], data["regs"][11],
                #                      data["regs"][12], data["regs"][13], data["regs"][14],
                #                      data["regs"][15]))
                #              print("Memory in emulator:")
                #              print("%04x: %s" % (addr, " ".join(["%02x" % cpu.state.read_memory(x, 1) for x in range(addr, addr + 16)])))
                #              print("Memory in trace:")
                #              print("%04x: %s" % (addr, " ".join(["%02x" % x for x in bytes])))
                #              raise RuntimeError("Mismatch between memory contents at address 0x%04x in tracefile line %d" % (addr + i, linenum))
#                print("CPU:")
#                print(str(cpu))
#                print("Trace:")
#                print(("pc  %04x    sp  %04x    sr  %04x    cg  %04x\n" + \
#                     "r4  %04x    r5  %04x    r6  %04x    r7  %04x\n" + \
#                     "r8  %04x    r9  %04x    r10 %04x    r11 %04x\n" + \
#                     "r12 %04x    r13 %04x    r14 %04x    r15 %04x\n") \
#                     % (data["regs"][0], data["regs"][1], data["regs"][2], 
#                        data["regs"][3], data["regs"][4], data["regs"][5],
#                        data["regs"][6], data["regs"][7], data["regs"][8],
#                        data["regs"][9], data["regs"][10], data["regs"][11],
#                        data["regs"][12], data["regs"][13], data["regs"][14], 
#                        data["regs"][15]))
                # cpu.step(True)
                cpu.step(True)
                if cpu.state.is_cpuoff():
                    sys.exit(0)
                lastinst = data["insn"]
                linenum += 1
    else:
        while True:
            cpu.step(True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type = str, default = None, help = "Program image as SREC file")
    parser.add_argument("--verify", type = str, default = None, help = "Verify execution against given web trace")
    parser.add_argument("--restore", type = str, default = None, help = "Restore system snapshot from JSON file")
    input_group = parser.add_argument_group("input")
    input_group.add_argument("--input", type = str, default = "", help = "Input to provide to the lock")
    input_group.add_argument("--hex", action = "store_true", default = False, help = "Input is hex-encoded")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    code = main(parse_args(), os.environ)
    if code:
        sys.exit(code)
	
