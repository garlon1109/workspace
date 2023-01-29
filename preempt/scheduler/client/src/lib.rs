#![feature(once_cell, c_unwind)]

use std::sync::Once;

use log::trace;

// This module is only for cbindgen generation and should not be used otherwise.
#[doc(hidden)]
pub mod types {
    #![allow(non_camel_case_types)]
    #[repr(C)]
    pub struct CUctx_st {
        _unused: [u8; 0],
    }
    #[repr(C)]
    pub struct CUstream_st {
        _unused: [u8; 0],
    }
    pub type CUcontext = *mut CUctx_st;
    pub type CUstream = *mut CUstream_st;

    #[test]
    fn test_size_align() {
        assert_eq!(
            std::mem::align_of::<CUcontext>(),
            std::mem::align_of::<reef_sys::CUcontext>()
        );
        assert_eq!(
            std::mem::align_of::<CUstream>(),
            std::mem::align_of::<reef_sys::CUstream>()
        );
        assert_eq!(
            std::mem::size_of::<CUcontext>(),
            std::mem::size_of::<reef_sys::CUcontext>()
        );
        assert_eq!(
            std::mem::size_of::<CUstream>(),
            std::mem::size_of::<reef_sys::CUstream>()
        );
        assert_eq!(
            std::mem::size_of::<CUcontext>(),
            std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<CUstream>(),
            std::mem::size_of::<usize>()
        );
    }
}

mod state;

/// called when a new CUDA context is created.
#[no_mangle]
pub extern "C" fn onContextInit(ctx: reef_sys::CUcontext) {
    static INIT: Once = Once::new();

    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();
    trace!("onContextInit: {ctx:?}");
    INIT.call_once(|| state::start_thread());
}

/// called when a new CUDA stream is created.
#[no_mangle]
pub extern "C" fn onStreamCreate(stream: reef_sys::CUstream) {
    trace!("onStreamCreate: {stream:?}");
    reef_sys::enable_preemption(stream);
    state::send_event(state::Event::StreamCreate(stream as usize));
}

/// called when a new CUDA stream is destroyed.
#[no_mangle]
pub extern "C" fn onStreamDestroy(stream: reef_sys::CUstream) {
    trace!("onStreamDestroy: {stream:?}");
    reef_sys::disable_preemption(stream);
    state::send_event(state::Event::StreamDestroy(stream as usize));
}

/// called when an empty PreemptableStream gets a new command.
#[no_mangle]
pub extern "C" fn onNewStreamCommand(stream: reef_sys::CUstream) {
    trace!("onNewStreamCommand: {stream:?}");
    state::send_event(state::Event::NewStreamCommand(stream as usize));
}

/// called when an non-empty PreemptableStream completes the last command.
#[no_mangle]
pub extern "C" fn onStreamSynchronized(stream: reef_sys::CUstream) {
    trace!("onStreamSynchronized: {stream:?}");
    state::send_event(state::Event::StreamSynchronized(stream as usize));
}
