use std::{collections::HashMap, os::unix::net::SocketAddr, sync::Arc};

use log::{trace, warn};
use thiserror::Error;
use transport::{
    msg::{ClientMessage, ServerMessage},
    socket::ServerEndpoint,
};

pub struct Scheduler {
    endpoint: Arc<ServerEndpoint>,
    clients: HashMap<u32, SocketAddr>,
}

#[derive(Debug, Error)]
#[error("Client disconnected")]
pub struct ClientDisconnected;

impl From<std::io::Error> for ClientDisconnected {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::ConnectionRefused {
            ClientDisconnected
        } else {
            panic!("Unexpected error: {}", e);
        }
    }
}

impl Scheduler {
    pub fn stop_ctx(&self, id: u32) -> Result<(), ClientDisconnected> {
        trace!("Stop context {}", id);
        self.endpoint
            .send(&ServerMessage::StopCtx, &self.clients[&id])?;
        Ok(())
    }
    pub fn resume_ctx(&self, id: u32) -> Result<(), ClientDisconnected> {
        trace!("Resume context {}", id);
        self.endpoint
            .send(&ServerMessage::ResumeCtx, &self.clients[&id])?;
        Ok(())
    }
}

/// A schudule policy
#[allow(unused_variables)]
pub trait Policy {
    fn on_ctx_create(&mut self, pid: u32, scheduler: &Scheduler) {}
    fn on_ctx_ready(&mut self, pid: u32, scheduler: &Scheduler) {}
    fn on_ctx_idle(&mut self, pid: u32, scheduler: &Scheduler) {}
    fn on_ctx_release(&mut self, pid: u32, scheduler: &Scheduler) {}
    fn on_set_priority(&mut self, pid: u32, priority: i32, scheduler: &Scheduler) {}
    fn get_priority(&self, pid: u32) -> i32 {
        0
    }
}

pub fn enter_server(mut policy: impl Policy) {
    let server = Arc::new(ServerEndpoint::new().unwrap());

    let mut scheduler = Scheduler {
        endpoint: server.clone(),
        clients: HashMap::new(),
    };

    loop {
        let (msg, addr) = server.recv().unwrap();
        trace!("received message {msg:?}");

        match msg {
            ClientMessage::CtxCreated(pid) => {
                scheduler.clients.insert(pid, addr);
                policy.on_ctx_create(pid, &scheduler);
            }
            ClientMessage::CtxReady(pid) => {
                scheduler.clients.insert(pid, addr);
                policy.on_ctx_ready(pid, &scheduler);
            }
            ClientMessage::CtxIdle(pid) => {
                scheduler.clients.insert(pid, addr);
                policy.on_ctx_idle(pid, &scheduler);
            }
            ClientMessage::CtxRelease(pid) => {
                scheduler.clients.insert(pid, addr);
                policy.on_ctx_release(pid, &scheduler);
            }
            ClientMessage::SetCtxPriority { pid, priority } => {
                policy.on_set_priority(pid, priority, &scheduler);
            }
            ClientMessage::GetCtxPriority(pid) => {
                let priority = policy.get_priority(pid);
                server
                    .send(&ServerMessage::CtxPriority { pid, priority }, &addr)
                    .unwrap_or_else(|e| warn!("Failed to send priority: {e}"));
            }
        }
    }
}
