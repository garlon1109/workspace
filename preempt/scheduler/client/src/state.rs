use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
    thread,
};

use crossbeam_channel::{unbounded, Receiver, Sender};
use log::trace;
use transport::{
    msg::{ClientMessage, ServerMessage},
    socket::ClientEndpoint,
};

pub struct ClientState {
    endpoint: Arc<ClientEndpoint>,
    stream_ready: HashMap<usize, bool>,
    preempted: bool,
}

impl ClientState {
    pub fn new() -> Self {
        let endpoint = ClientEndpoint::new_connected().unwrap();
        let pid = std::process::id();
        endpoint.send(&ClientMessage::CtxCreated(pid)).unwrap();
        Self {
            endpoint: Arc::new(endpoint),
            stream_ready: Default::default(),
            preempted: false,
        }
    }

    pub fn stream_created(&mut self, id: usize) {
        self.set_inactive(id);
        if self.preempted {
            reef_sys::preempt(id as reef_sys::CUstream, false);
        }
    }

    pub fn stream_distroyed(&mut self, id: usize) {
        self.stream_ready.remove(&id);
    }

    pub fn set_active(&mut self, id: usize) {
        // if all streams are previously inactive, send CtxReady
        if self.stream_ready.values().all(|&v| !v) {
            let pid = std::process::id();
            self.endpoint.send(&ClientMessage::CtxReady(pid)).unwrap();
        }
        self.stream_ready.insert(id, true);
    }

    pub fn set_inactive(&mut self, id: usize) {
        let previous = self.stream_ready.insert(id, false);
        // if all streams are now inactive, send CtxIdle
        if previous == Some(true) && self.stream_ready.values().all(|&v| !v) {
            let pid = std::process::id();
            self.endpoint.send(&ClientMessage::CtxIdle(pid)).unwrap();
        }
    }
}

pub enum Event {
    StreamCreate(usize),
    StreamDestroy(usize),
    NewStreamCommand(usize),
    StreamSynchronized(usize),
    StopCtx,
    ResumeCtx,
}

// channel as an inbox for events
static EVENT_CHANNEL: OnceLock<Sender<Event>> = OnceLock::new();

pub fn send_event(event: Event) {
    let sender = EVENT_CHANNEL.get().unwrap();
    sender.send(event).unwrap();
}

pub fn start_thread() {
    let (tx, rx) = unbounded();
    EVENT_CHANNEL.set(tx).unwrap();
    let state = ClientState::new();
    let endpoint = state.endpoint.clone();
    thread::Builder::new()
        .name("receiver".to_string())
        .spawn(move || recv_msg(endpoint))
        .unwrap();
    thread::Builder::new()
        .name("sched client".to_string())
        .spawn(move || main_loop(rx, state))
        .unwrap();
}

fn main_loop(rx: Receiver<Event>, mut state: ClientState) -> ! {
    loop {
        match rx.recv().unwrap() {
            Event::StreamCreate(id) => {
                state.stream_created(id);
            }
            Event::StreamDestroy(id) => {
                state.stream_distroyed(id);
            }
            Event::NewStreamCommand(id) => {
                state.set_active(id);
            }
            Event::StreamSynchronized(id) => {
                state.set_inactive(id);
            }
            Event::StopCtx => {
                state.preempted = true;
                for &id in state.stream_ready.keys() {
                    reef_sys::preempt(id as reef_sys::CUstream, false);
                }
            }
            Event::ResumeCtx => {
                state.preempted = false;
                for &id in state.stream_ready.keys() {
                    reef_sys::restore(id as reef_sys::CUstream);
                }
            }
        }
    }
}

fn recv_msg(endpoint: Arc<ClientEndpoint>) -> ! {
    loop {
        match endpoint.recv().unwrap() {
            ServerMessage::StopCtx => {
                trace!("Received StopCtx");
                send_event(Event::StopCtx);
            }
            ServerMessage::ResumeCtx => {
                trace!("Received ResumeCtx");
                send_event(Event::ResumeCtx);
            }
            _ => {}
        }
    }
}
