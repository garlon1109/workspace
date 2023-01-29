use std::os::unix::net::UnixDatagram;
use std::{io::Result, os::unix::net::SocketAddr};

use log::trace;
use rkyv::AlignedBytes;

use crate::msg::{ClientMessage, ServerMessage};

const SERVER_SOCK_ADDR: &str = "/tmp/reef-server.sock";

pub struct ClientEndpoint {
    socket: UnixDatagram,
}

impl ClientEndpoint {
    pub fn new_connected() -> Result<Self> {
        let pid = std::process::id();
        let socket = UnixDatagram::bind(format!("/tmp/reef-client-{}.sock", pid))?;
        socket.connect(SERVER_SOCK_ADDR)?;
        Ok(ClientEndpoint { socket })
    }

    pub fn send(&self, msg: &ClientMessage) -> Result<()> {
        trace!("Sending message: {:?}", msg);
        let mut buf = [0; ClientMessage::SIZE];
        msg.to_bytes(&mut buf);
        self.socket.send(&buf)?;
        Ok(())
    }

    pub fn recv(&self) -> Result<ServerMessage> {
        let mut buf = AlignedBytes::default();
        self.socket.recv(&mut *buf)?;
        Ok(ServerMessage::from_bytes(&buf))
    }
}

pub struct ServerEndpoint {
    socket: UnixDatagram,
}

impl ServerEndpoint {
    pub fn new() -> Result<Self> {
        std::fs::remove_file(SERVER_SOCK_ADDR).ok();
        let socket = UnixDatagram::bind(SERVER_SOCK_ADDR)?;
        Ok(ServerEndpoint { socket })
    }

    pub fn recv(&self) -> Result<(ClientMessage, SocketAddr)> {
        let mut buf = AlignedBytes::default();
        let (_, src) = self.socket.recv_from(&mut *buf)?;
        Ok((ClientMessage::from_bytes(&buf), src))
    }

    pub fn send(&self, msg: &ServerMessage, addr: &SocketAddr) -> Result<()> {
        let mut buf = [0; ServerMessage::SIZE];
        msg.to_bytes(&mut buf);
        self.socket.send_to_addr(&buf, addr)?;
        Ok(())
    }
}
