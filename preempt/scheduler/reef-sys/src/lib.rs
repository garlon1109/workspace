mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use ffi::{CUcontext, CUstream};

pub fn preempt(stream: CUstream, synchronize: bool) {
    let ret = unsafe { ffi::reef_preempt(stream, synchronize) };
    assert_eq!(ret, ffi::reef_RFresult_REEF_SUCCESS);
}

pub fn restore(stream: CUstream) {
    let ret = unsafe { ffi::reef_restore(stream) };
    assert_eq!(ret, ffi::reef_RFresult_REEF_SUCCESS);
}

pub fn enable_preemption(stream: CUstream) {
    let config = ffi::reef_RFconfig {
        queueSize: 1024,
        batchSize: 512,
        taskTimeout: 1000,
        preemptLevel: ffi::reef_PreemptLevel_PreemptHostQueue,
    };
    let ret = unsafe { ffi::reef_enablePreemption(stream, config) };
    assert_eq!(ret, ffi::reef_RFresult_REEF_SUCCESS);
}

pub fn disable_preemption(stream: CUstream) {
    let ret = unsafe { ffi::reef_disablePreemption(stream) };
    assert_eq!(ret, ffi::reef_RFresult_REEF_SUCCESS);
}

pub fn cu_ctx_set_current(ctx: CUcontext) {
    let ret = unsafe { ffi::cuCtxSetCurrent(ctx) };
    assert_eq!(ret, ffi::cudaError_enum_CUDA_SUCCESS);
}
