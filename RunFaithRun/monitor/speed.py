from ctypes import (
    pointer, sizeof, windll, create_string_buffer,
    c_ulong, byref, GetLastError, c_bool, WinError,
    c_size_t, c_long, c_longlong, c_float, POINTER
)
import struct
import numpy as np

psapi       = windll.psapi
kernel32    = windll.kernel32

from ctypes import wintypes as w
RPM = kernel32.ReadProcessMemory
RPM.argtypes = [w.HANDLE,w.LPCVOID,w.LPVOID,c_size_t,POINTER(c_size_t)]
RPM.restype = w.BOOL

# partial credit: memorpy 
def proclist():
    processes=[]
    arr = c_ulong * 256
    lpidProcess= arr()
    cb = sizeof(lpidProcess)
    cbNeeded = c_ulong()
    hModule = c_ulong()
    count = c_ulong()
    modname = create_string_buffer(100)
    PROCESS_QUERY_INFORMATION = 0x0400
    PROCESS_VM_READ = 0x0010

    psapi.EnumProcesses(byref(lpidProcess), cb, byref(cbNeeded))
    nReturned = cbNeeded.value//sizeof(c_ulong())

    pidProcess = [i for i in lpidProcess][:nReturned]
    for pid in pidProcess:
        proc={ "pid": int(pid) }
        hProcess = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if hProcess:
            psapi.EnumProcessModules(hProcess, byref(hModule), sizeof(hModule), byref(count))
            psapi.GetModuleBaseNameA(hProcess, hModule.value, modname, sizeof(modname))
            proc["name"]=modname.value
            kernel32.CloseHandle(hProcess)
        processes.append(proc)
    return processes

def processes_from_name(processName):
    processes = []
    for process in proclist():
        if processName == process.get("name", ''):
            processes.append(process)

    if len(processes) > 0:
        return processes

def read(handler, address, ctype):
    buf = create_string_buffer(sizeof(ctype))
    ptr = c_size_t(0)
    res = RPM(
            handler, address,
            buf, sizeof(ctype), byref(ptr))
    if ctype==c_longlong:
        return struct.unpack("<q", buf)[0]
    elif ctype==c_float:
        return struct.unpack("<f", buf)[0]

def get_MEC_address(handler, base=0x140000000):
    tmp = read(handler, base+0x02578A68, c_longlong)
    print (tmp)
    tmp = read(handler, tmp+0x70, c_longlong)
    print (tmp)
    tmp = read(handler, tmp+0x98, c_longlong)
    print (tmp)
    tmp = read(handler, tmp+0x238, c_longlong)
    print (tmp)
    tmp = read(handler, tmp+0x18, c_longlong)
    print (tmp)

    y_ptr = tmp + 0x22d4
    x_ptr = tmp + 0x22d0
    z_ptr = tmp + 0x22d8
    return x_ptr, y_ptr, z_ptr

# print('read res:', res)
# print('buf:', buf.value)
# print(struct.unpack("<q", buf))[0]
#   print('last error:', GetLastError())
#   print('ptr:', ptr.value)
#   print('buf.raw:', buf.raw)

class Speedometer(object):
    def __init__(self):
        MCE = processes_from_name(b"MirrorsEdgeCatalyst.exe")
        pid = MCE[0]['pid']

        self.h_process = h_process = kernel32.OpenProcess(0x001F0FFF, 0, pid)

        print(read(h_process, 0x02578A68, c_longlong))
        print(read(h_process, 0x142578A68, c_longlong))
        self.xyz_ptrs = get_MEC_address(h_process)
        print("{:x}|{:x}|{:x}".format(*self.xyz_ptrs))

    def get_position(self):
        pos = [read(self.h_process, ptr, c_float) for ptr in self.xyz_ptrs]
        return np.array(pos)

    def get_speed(self):
        cur_pos = self.get_position()
        if self.last_pos is None:
            self.last_pos = cur_pos 
            return np.array([0, 0, 0])
        else:
            speed = cur_pos-self.last_pos
            self.last_pos = cur_pos 
            return speed

    def reset(self):
        self.last_pos = None

if __name__ == '__main__':
    MCE = processes_from_name(b"MirrorsEdgeCatalyst.exe")
    pid = MCE[0]['pid']

    h_process = kernel32.OpenProcess(0x001F0FFF, 0, pid)

    print(read(h_process, 0x02578A68, c_longlong))
    print(read(h_process, 0x142578A68, c_longlong))
    xyz_ptrs = get_MEC_address(h_process)
    print("{:x}|{:x}|{:x}".format(*xyz_ptrs))
    from time import sleep
    while True:
        sleep(0.5)
        print([read(h_process, ptr, c_float) for ptr in xyz_ptrs])
    
    kernel32.CloseHandle(h_process)
