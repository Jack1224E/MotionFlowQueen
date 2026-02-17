import ctypes
import numpy as np
import torch
import sys

# === Constants & Offsets (from offset_finder) ===
# NV_OF_INIT_PARAMS size 48
# offsets: width=0, height=4, outGridSize=8, mode=16, perfLevel=20, enableExternalHints=24, enableOutputCost=28, hPrivData=32
# NV_OF_CUDA_API_FUNCTION_LIST size 96
# offsets: nvOFInit=8, nvOFExecute=56, nvOFDestroy=72, nvOFGetCaps=88
# NV_OF_EXECUTE_INPUT_PARAMS size 56
# offsets: inputFrame=0, referenceFrame=8, disableTemporalHints=24
# NV_OF_EXECUTE_OUTPUT_PARAMS size 24
# offsets: outputBuffer=0, outputCostBuffer=8

NV_OF_SUCCESS = 0
NV_OF_MODE_OPTICALFLOW = 1
NV_OF_PERF_LEVEL_SLOW = 5
NV_OF_PERF_LEVEL_MEDIUM = 10
NV_OF_PERF_LEVEL_FAST = 20
NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 = 4
NV_OF_BUFFER_USAGE_INPUT = 1
NV_OF_BUFFER_USAGE_OUTPUT = 2
NV_OF_BUFFER_FORMAT_NV12 = 2
NV_OF_BUFFER_FORMAT_ABGR8 = 3
NV_OF_BUFFER_FORMAT_SHORT2 = 5
NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2

# === Ctypes Structures ===
class NV_OF_INIT_PARAMS(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("outGridSize", ctypes.c_int),
        ("hintGridSize", ctypes.c_int), # inferred (4 bytes)
        ("mode", ctypes.c_int),         # offset 16 confirmed
        ("perfLevel", ctypes.c_int),    # offset 20 confirmed
        ("enableExternalHints", ctypes.c_int), # 24
        ("enableOutputCost", ctypes.c_int),    # 28
        ("hPrivData", ctypes.c_void_p),        # 32
        ("disparityRange", ctypes.c_int),      # inferred
        ("enableRoi", ctypes.c_int)            # inferred
    ]

# Function pointer types
class NvOFHandle(ctypes.c_void_p): pass
class NvOFGPUBufferHandle(ctypes.c_void_p): pass
class CUcontext(ctypes.c_void_p): pass
class CUstream(ctypes.c_void_p): pass

PFNNVCREATEOPTICALFLOWCUDA = ctypes.CFUNCTYPE(ctypes.c_int, CUcontext, ctypes.POINTER(NvOFHandle))
PFNNVOFINIT = ctypes.CFUNCTYPE(ctypes.c_int, NvOFHandle, ctypes.POINTER(NV_OF_INIT_PARAMS))
PFNNVOFEXECUTE = ctypes.CFUNCTYPE(ctypes.c_int, NvOFHandle, ctypes.c_void_p, ctypes.c_void_p) # using void_p for flexible input/output params
PFNNVOFDESTROY = ctypes.CFUNCTYPE(ctypes.c_int, NvOFHandle)
PFNNVOFSETIOCUDASTREAMS = ctypes.CFUNCTYPE(ctypes.c_int, NvOFHandle, CUstream, CUstream)

class NV_OF_CUDA_API_FUNCTION_LIST(ctypes.Structure):
    _fields_ = [
        ("nvCreateOpticalFlowCuda", PFNNVCREATEOPTICALFLOWCUDA),
        ("nvOFInit", PFNNVOFINIT),
        ("nvOFCreateGPUBufferCuda", ctypes.c_void_p),
        ("nvOFGPUBufferGetCUarray", ctypes.c_void_p),
        ("nvOFGPUBufferGetCUdeviceptr", ctypes.c_void_p),
        ("nvOFGPUBufferGetStrideInfo", ctypes.c_void_p),
        ("nvOFSetIOCudaStreams", PFNNVOFSETIOCUDASTREAMS),
        ("nvOFExecute", PFNNVOFEXECUTE), # offset 56 matches layout? 
        # offset_finder said nvOFExecute is at 56.
        # 0: Create (8 bytes)
        # 8: Init (8 bytes)
        # 16: CreateBuffer (8 bytes)
        # 24: GetCUarray (8 bytes)
        # 32: GetCUdeviceptr (8 bytes)
        # 40: GetStride (8 bytes)
        # 48: SetIOStreams (8 bytes)
        # 56: Execute (8 bytes) - MATCHES!
        ("nvOFDestroyGPUBufferCuda", ctypes.c_void_p),
        ("nvOFDestroy", PFNNVOFDESTROY),
        ("nvOFGetLastError", ctypes.c_void_p),
        ("nvOFGetCaps", ctypes.c_void_p)
    ]

class NV_OF_EXECUTE_INPUT_PARAMS(ctypes.Structure):
    _fields_ = [
        ("inputFrame", NvOFGPUBufferHandle),       # 0
        ("referenceFrame", NvOFGPUBufferHandle),   # 8
        ("externalHints", NvOFGPUBufferHandle),    # 16
        ("disableTemporalHints", ctypes.c_int),    # 24
        ("padding", ctypes.c_uint32),              # 28
        ("hPrivData", ctypes.c_void_p),            # 32
        ("padding2", ctypes.c_uint32),             # 40
        ("numRois", ctypes.c_uint32),              # 44
        ("roiData", ctypes.c_void_p)               # 48
    ]
    # Check size: 0+8=8, +8=16, +8=24, +4=28, +4=32, +8=40, +4=44, +4=48, +8=56. MATCHES offset_finder (56).

class NV_OF_EXECUTE_OUTPUT_PARAMS(ctypes.Structure):
    _fields_ = [
        ("outputBuffer", NvOFGPUBufferHandle),     # 0
        ("outputCostBuffer", NvOFGPUBufferHandle), # 8
        ("hPrivData", ctypes.c_void_p)             # 16
    ]
    # Check size: 0+8=8, +8=16, +8=24. MATCHES offset_finder (24).


# === Main Smoke Test ===
def main():
    print("=== NVOFA Real-Hardware Smoke Test ===")
    
    # 1. Load Libraries
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        libnvofa = ctypes.CDLL("libnvidia-opticalflow.so.1")
    except OSError as e:
        print(f"[!] Failed to load libraries: {e}")
        return

    # Define CUDA functions
    cuCtxGetCurrent = libcuda.cuCtxGetCurrent
    cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
    cuCtxGetCurrent.restype = int
    
    cuCtxPushCurrent = libcuda.cuCtxPushCurrent
    cuCtxPushCurrent.argtypes = [CUcontext]
    cuCtxPushCurrent.restype = int
    
    cuCtxPopCurrent = libcuda.cuCtxPopCurrent
    cuCtxPopCurrent.argtypes = [ctypes.POINTER(CUcontext)]
    cuCtxPopCurrent.restype = int
    
    cuStreamCreate = libcuda.cuStreamCreate
    cuStreamCreate.argtypes = [ctypes.POINTER(CUstream), ctypes.c_uint]
    cuStreamCreate.restype = int

    cuStreamDestroy = libcuda.cuStreamDestroy
    cuStreamDestroy.argtypes = [CUstream]
    cuStreamDestroy.restype = int

    cuStreamSynchronize = libcuda.cuStreamSynchronize
    cuStreamSynchronize.argtypes = [CUstream]
    cuStreamSynchronize.restype = int

    # 2. Initialize PyTorch (creates primary context)
    if not torch.cuda.is_available():
        print("[!] PyTorch CUDA not available")
        return
    
    device = torch.device('cuda:0')
    dummy = torch.zeros(1).to(device) # Force context creation
    print("[+] PyTorch Context Initialized")

    # 3. Get Current Context
    current_ctx = CUcontext()
    ret = cuCtxGetCurrent(ctypes.byref(current_ctx))
    if ret != 0 or not current_ctx:
        print(f"[!] Failed to get current CUDA context. Ret: {ret}")
        return
    print(f"[+] Retrieved PyTorch Context: {current_ctx}")

    # 4. Create API Instance
    api_func_list = NV_OF_CUDA_API_FUNCTION_LIST()
    # NV_OF_API_VERSION = (Major << 4) | Minor. For 2.0, this is (2 << 4) | 0 = 32.
    api_ver = (2 << 4) | 0
    ret = libnvofa.NvOFAPICreateInstanceCuda(api_ver, ctypes.byref(api_func_list)) # Ver 2.0
    if ret != 0:
        print(f"[!] NvOFAPICreateInstanceCuda failed. Ret: {ret}")
        return
    print("[+] NVOFA API Instance Created")

    # 5. Create NVOFA Object
    hOf = NvOFHandle()
    
    # PUSH CONTEXT
    cuCtxPushCurrent(current_ctx)
    print("    [Context Pushed]")
    
    try:
        # Try passing None (0) to use current context?
        # ret = api_func_list.nvCreateOpticalFlowCuda(current_ctx, ctypes.byref(hOf))
        ret = api_func_list.nvCreateOpticalFlowCuda(None, ctypes.byref(hOf))
        if ret != 0:
            print(f"[!] nvCreateOpticalFlowCuda failed. Ret: {ret}")
            # Restore context check
            if ret == 2: print("    [!] Error 2: UNSUPPORTED_DEVICE. Likely GTX 1650 specific.")
            return
        print(f"[+] NVOFA Handle Created: {hOf}")

        # 6. Initialize Session
        width, height = 640, 384
        init_params = NV_OF_INIT_PARAMS()
        init_params.width = width
        init_params.height = height
        init_params.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4
        init_params.mode = NV_OF_MODE_OPTICALFLOW
        init_params.perfLevel = NV_OF_PERF_LEVEL_MEDIUM
        init_params.enableExternalHints = 0
        init_params.enableOutputCost = 0
        init_params.enableRoi = 0

        ret = api_func_list.nvOFInit(hOf, ctypes.byref(init_params))
        if ret != 0:
            print(f"[!] nvOFInit failed. Ret: {ret}")
            return
        print("[+] NVOFA Session Initialized")

        # 7. Create Non-Blocking Stream
        stream = CUstream()
        # CU_STREAM_NON_BLOCKING = 1
        cuStreamCreate(ctypes.byref(stream), 1)
        print(f"[+] Created Non-Blocking Stream: {stream}")

        # Set IO Streams
        api_func_list.nvOFSetIOCudaStreams(hOf, stream, stream)
        print("[+] IO Streams Set")
        
        # 8. Prepare Buffers (Using PyTorch Tensors as `DevicePtr`)
        # Input: ABGR8 (4 bytes per pixel)
        # But wait, user mentioned "Raw uint8 ABGR or NV12".
        # Let's use ABGR8 for simplicity. PyTorch default is float32. We need ByteTensor.
        # Shape: (1, 4, H, W)? No, ABGR is packed. (1, H, W, 4).
        
        img0 = torch.zeros((height, width, 4), dtype=torch.uint8, device=device)
        img1 = torch.ones((height, width, 4), dtype=torch.uint8, device=device) * 255 # White
        
        # Create Output Buffer
        # Grid 4x4. Output size: Width/4, Height/4.
        out_w = (width + 3) // 4
        out_h = (height + 3) // 4
        # Format SHORT2 (2x int16). 4 bytes per vector.
        output_flow = torch.zeros((out_h, out_w, 2), dtype=torch.int16, device=device)
        
        # Wrap pointers
        # NvOFGPUBufferHandle is a pointer (void*).
        # We need to pass it as `inputFrame`. But `inputFrame` expects a Handle that we created?
        # WAIT. NVOFA typically manages its own buffers via `nvOFCreateGPUBufferCuda` OR accepts Raw Device Ptrs?
        # nvOpticalFlowCuda.h says:
        # "If ::NV_OF_INIT_PARAMS::mode is ::NV_OF_MODE_OPTICALFLOW, this specifies the handle to the buffer containing the input frame."
        # AND "Supported buffer types... NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR".
        # If we use `nvOFCreateGPUBufferCuda` with `CUDEVICEPTR`, we wrap our ptr.
        # OR does Execute accept raw pointers if configured?
        # Usually we MUST register the buffer or create a wrapper handle.
        # Let's use `nvOFCreateGPUBufferCuda` to wrap our PyTorch ptrs.
        
        def create_buffer_handle(tensor, usage, format_):
            handle = NvOFGPUBufferHandle()
            desc = ctypes.Structure() # Placeholder? No need def.
            # We need NV_OF_BUFFER_DESCRIPTOR
             
            class NV_OF_BUFFER_DESCRIPTOR(ctypes.Structure):
                _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                            ("bufferUsage", ctypes.c_int), ("bufferFormat", ctypes.c_int)]
            
            bdesc = NV_OF_BUFFER_DESCRIPTOR()
            bdesc.width = tensor.shape[1] # W
            bdesc.height = tensor.shape[0] # H
            bdesc.bufferUsage = usage
            bdesc.bufferFormat = format_
            
            # We need to pass the PTR?
            # Creating a buffer 'handle' doesn't take the ptr yet?
            # `nvOFCreateGPUBufferCuda` takes `bufferType`. If `CUDEVICEPTR`, where do we pass the ptr?
            # Usually we don't pass the ptr at creation for `CUDEVICEPTR`?
            # Wait, `nvOFCreateGPUBufferCuda` returns a Handle.
            # Then how do we associate the Ptr?
            # RIF reqs: For `CUDEVICEPTR`, we pass the device ptr inside the Handle? No.
            # Actually, for `CUDEVICEPTR` mode, `NvOFGPUBufferHandle` IS the device pointer?
            # Or we cast usage?
            # Let's check docs/header.
            # Header says: `NvOFGPUBufferHandle` is `struct NvOFGPUBufferHandle_st*`.
            # `nvOFGPUBufferGetCUdeviceptr` returns the ptr.
            # It seems we might need to *register* it?
            # Actually, often `inputFrame` in ExecuteParams can be cast from CUdeviceptr if using a specific interface?
            # But the signature demands `NvOFGPUBufferHandle`.
            # If we assume we must create a handle...
            # The SDK sample `AppOFCuda` creates buffers.
            # But we want to use PyTorch buffers (zero copy).
            # The "Paranoia" plan didn't specify this detail, but "Part 5" size "Allocate PyTorch output...".
            # This implies zero copy.
            # If we can't zero-copy easily without registering, we might crash.
            # Let's assume we can simply *cast* the pointer to `NvOFGPUBufferHandle` IF the driver supports it?
            # NO, that's dangerous.
            
            # Alternative: `nvOFCreateGPUBufferCuda` creates an opaque handle.
            # But we have existing data.
            # Maybe we skip the wrapper for now and just pass `ctypes.c_void_p(img0.data_ptr())`?
            # If `nvOFExecute` expects `NvOFGPUBufferHandle`, passing a raw ptr is type mismatch.
            # BUT: In many NVIDIA APIs, if you pass `CUdeviceptr`, it works?
            # The header defines `NvOFGPUBufferHandle` as `struct*`.
            # It is likely we MUST create a registered buffer.
            # But `nvOFCreateGPUBufferCuda` with `NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR`...
            # Does it take the ptr as input? The signature is `(hOf, desc, bufferType, *hHandle)`.
            # It does NOT take the ptr.
            # So `nvOFCreateGPUBufferCuda` allocates NEW memory?
            # "This function creates ... resource ... for specified cuda bufferType."
            # If type is `CUDEVICEPTR`, maybe it allocates a container?
            # Then how do we set the value?
            # We probably can't use `CUDEVICEPTR` type for *importing* external pointers easily without a Register function?
            # Wait, `nvOFGPUBufferGetCUdeviceptr` retrieves the ptr.
            
            # Let's try the safest path: Let NVOFA allocate the input/output buffers, copy from PyTorch, Execute, Copy back.
            # It's less efficient but SAFER for a smoke test.
            # Validate HW works first. Optimization later.
            
            # So:
            # 1. Create Input Buffers via NVOFA.
            # 2. Get their CUdeviceptr.
            # 3. MemcpyAsync from PyTorch to NVOFA Input.
            # 4. Execute.
            # 5. MemcpyAsync from NVOFA Output to PyTorch.
            pass
        
        # Helper for buffer creation
        class NV_OF_BUFFER_DESCRIPTOR(ctypes.Structure):
                _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
                            ("bufferUsage", ctypes.c_int), ("bufferFormat", ctypes.c_int)]
        
        def alloc_nvofa_buf(w, h, usage, fmt):
            desc = NV_OF_BUFFER_DESCRIPTOR(w, h, usage, fmt)
            hBuf = NvOFGPUBufferHandle()
            res = api_func_list.nvOFCreateGPUBufferCuda(hOf, ctypes.byref(desc), NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, ctypes.byref(hBuf))
            if res != 0: raise RuntimeError(f"Alloc failed {res}")
            return hBuf

        print("[.] Allocating Buffers...")
        inBuf0 = alloc_nvofa_buf(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8)
        inBuf1 = alloc_nvofa_buf(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_ABGR8)
        outBuf = alloc_nvofa_buf(out_w, out_h, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2)
        print("[+] Buffers Allocated")

        # Get Pointers
        ptr0 = api_func_list.nvOFGPUBufferGetCUdeviceptr(inBuf0)
        ptr1 = api_func_list.nvOFGPUBufferGetCUdeviceptr(inBuf1)
        ptrOut = api_func_list.nvOFGPUBufferGetCUdeviceptr(outBuf)
        
        # Copy Data (PyTorch -> NVOFA Ptr)
        # Using torch.cuda.Memcpy? Or just tensor.copy?
        # We can wrap the NVOFA ptr as a PyTorch tensor!
        # This is the "Zero Copy" (sort of) approach for access.
        
        # Wrapper function
        def wrap_ptr(ptr, shape, dtype):
            # Hacky: create a tensor from a pointer.
            # Easier: Use cudaMemcpyAsync via ctypes or define it.
            # Or assume we can just copy.
            pass
        
        # Let's use `cudaMemcpy` from libcuda (driver api) or `cudaMemcpy2D`?
        # Simpler: Create a tensor from the pointer?
        # PyTorch doesn't expose `from_blob` with device ptr easily in older versions, but `as_strided`?
        # Or `cudart`?
        
        # Taking "Paranoia" approach: Use `cuMemcpyDtoDAsync`.
        cuMemcpyDtoDAsync = libcuda.cuMemcpyDtoDAsync
        cuMemcpyDtoDAsync.argtypes = [CUdeviceptr, CUdeviceptr, ctypes.c_size_t, CUstream]
        cuMemcpyDtoDAsync.restype = int
        
        # Size
        input_size = width * height * 4
        output_size = out_w * out_h * 4
        
        # Copy img0 -> ptr0
        cuMemcpyDtoDAsync(ptr0, img0.data_ptr(), input_size, stream)
        # Copy img1 -> ptr1
        cuMemcpyDtoDAsync(ptr1, img1.data_ptr(), input_size, stream)
        print("[.] Inputs Copied")
        
        # 9. Execute
        exec_in = NV_OF_EXECUTE_INPUT_PARAMS()
        exec_in.inputFrame = inBuf0
        exec_in.referenceFrame = inBuf1
        exec_in.disableTemporalHints = 1 # Deterministic!
        
        exec_out = NV_OF_EXECUTE_OUTPUT_PARAMS()
        exec_out.outputBuffer = outBuf
        
        ret = api_func_list.nvOFExecute(hOf, ctypes.byref(exec_in), ctypes.byref(exec_out))
        if ret != 0:
            print(f"[!] nvOFExecute failed. Ret: {ret}")
            return
        print("[+] Execution Submitted")
        
        # 10. Copy Back
        cuMemcpyDtoDAsync(output_flow.data_ptr(), ptrOut, output_size, stream)
        
        # Synchronize Stream
        cuStreamSynchronize(stream)
        print("[+] Stream Synchronized")
        
        # 11. Validate
        # Print first 4x4 vector
        flow_cpu = output_flow.cpu().numpy()
        print("First flow vector:", flow_cpu[0,0])
        
        # Destroy Buffers
        api_func_list.nvOFDestroyGPUBufferCuda(inBuf0)
        api_func_list.nvOFDestroyGPUBufferCuda(inBuf1)
        api_func_list.nvOFDestroyGPUBufferCuda(outBuf)
        
    finally:
        # Tear down
        if 'hOf' in locals() and hOf:
            api_func_list.nvOFDestroy(hOf)
            print("[+] NVOFA Destroyed")
        
        cuCtxPopCurrent(ctypes.byref(current_ctx))
        print("    [Context Popped]")
        
        if 'stream' in locals() and stream:
            cuStreamDestroy(stream)
            print("[+] Stream Destroyed")

if __name__ == "__main__":
    main()
