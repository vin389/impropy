import ctypes 
import numpy as np 

# Load library from shared library (e.g., DLL file)
impropycLib = ctypes.cdll.LoadLibrary( \
    './build-impropyc-Desktop_Qt_5_15_0_MSVC2019_64bit-Release/release/impropyc.dll')

# set argument types 
impropycLib.myCvMat.argTypes = (
    ctypes.POINTER(ctypes.c_uint8), \
    ctypes.c_int, \
    ctypes.c_int )
impropycLib.myCvMat.restype = ctypes.c_int 
def improMyCvMat(m, n):
    mat = np.zeros((m, n), dtype=np.uint8)
    pmat = mat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    impropycLib.myCvMat(pmat, m, n)
    return mat


    