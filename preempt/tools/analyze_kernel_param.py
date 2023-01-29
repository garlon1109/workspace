import json
import os
import subprocess
import sys
import re
import argparse

class KernelParam:

    fundamental_type_size = {
        ".s8": 1,
        ".s16": 2,
        ".s32": 4,
        ".s64": 8,
        ".u8": 1,
        ".u16": 2,
        ".u32": 4,
        ".u64": 8,
        ".f16": 2,
        ".f16x2": 4,
        ".f32": 4,
        ".f64": 8,
        ".b8": 1,
        ".b16": 2,
        ".b32": 4,
        ".b64": 8,
        ".pred": 1,
    }

    def __init__(self, type, size, cnt):
        self.type = type
        self.size = size
        self.cnt = cnt
        return

    def parse_str(param_str):
        parts = param_str.strip().split(" ")
        assert parts[0] == ".param", f"param_str: {param_str}"
        type = None
        find_type = False
        type_size = 0
        type_cnt = 1
        for part in parts:
            if part in KernelParam.fundamental_type_size:
                find_type = True
                type = part
                type_size = KernelParam.fundamental_type_size[type]
        assert find_type == True
        array_size = re.findall("\[\d*\]", parts[-1])
        if len(array_size) != 0:
            assert len(array_size) == 1
            type_cnt = int(array_size[0][1:-1])
        return KernelParam(type, type_size, type_cnt)
    
    def __repr__(self):
        return f"KernelParam{{type: {self.type}, size: {self.size}, cnt: {self.cnt}}}"


class KernelInfo:
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def max_param_size(self):
        return max([param.size * param.cnt for param in self.params])

    def __repr__(self) -> str:
        return f"KernelInfo{{name: {self.name}, params: {self.params}}}"

def dump_ptx(file):
    status, output = subprocess.getstatusoutput(f'cuobjdump -ptx {file}')
    if status != 0:
        print(f"cuobjdump returns {status}")
        exit(1)
    return output

def extract_kernels(ptx):
    p = re.compile(".entry (\w*)\(\n((\.param.*,?\n)*)")
    kernel_list = p.findall(ptx)
    res = []
    for kernel_raw in kernel_list:
        kname = kernel_raw[0]
        kparam = kernel_raw[1].strip().replace(",","").split("\n")
        try:
            kparam = [KernelParam.parse_str(s) for s in kparam]
            res.append(KernelInfo(kname, kparam))
        except:
            print(f"parse param error: {kernel_raw}")
        
    return res


def max_kernel_param_num(kernels):
    return max([len(kernel.params) for kernel in kernels])

def max_kernel_param_size(kernels):
    return max([kernel.max_param_size() for kernel in kernels])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cb", "--cubin", help="analyze cuda binary file")
    parser.add_argument("-ptx", "--ptx", help="analyze ptx file")
    parser.add_argument("-map", "--map", help="saved map file")
    args = parser.parse_args()
    if args.cubin == None and args.ptx == None:
        print("Must specify cubin or ptx file.")
        exit(1)
    ptx_raw = ""
    if args.cubin != None:
        ptx_raw = dump_ptx(args.cubin)
    else:
        ptx_file = open(args.ptx, "r")
        ptx_lines = ptx_file.readlines()
        ptx_lines = [l.strip() for l in ptx_lines]
        ptx_raw = "\n".join(ptx_lines)
        ptx_file.close()
    kernels = extract_kernels(ptx_raw)
    print(f"num kenerls: {len(kernels)}")
    if len(kernels) > 0:
        print(f"max_kernel_param_num: {max_kernel_param_num(kernels)}")
        print(f"max_kernel_param_size: {max_kernel_param_size(kernels)} bytes")

    if args.map != None:
        if os.path.exists(args.map):
            map_file = open(args.map, "r+")
            map = json.load(map_file)
            map_file.seek(0)
            map_file.truncate()
        else:
            map_file = open(args.map, "w+")
            map = {}

        for kernel in kernels:
            map[kernel.name] = [[param.cnt, param.type] for param in kernel.params]

        json.dump(map, map_file)
    