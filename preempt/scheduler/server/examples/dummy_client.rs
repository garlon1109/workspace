use std::{sync::Arc, thread, time::Duration};

use transport::{msg::ClientMessage, socket::ClientEndpoint};

fn main() {
    // Bind to /tmp/reef-client-<pid>.sock
    let endpoint = Arc::new(ClientEndpoint::new_connected().unwrap());
    thread::spawn({
        let endpoint = endpoint.clone();
        move || {
            loop {
                let msg = endpoint.recv().unwrap();
                println!("[Client] Received message: {:?}", msg);
            }
        }
    });
    // Send created message to SERVER_SOCK
    let pid = std::process::id();
    endpoint.send(&ClientMessage::CtxCreated(pid)).unwrap();

    std::thread::sleep(Duration::from_secs(10));
    endpoint.send(&ClientMessage::CtxReady(pid)).unwrap();

    std::thread::sleep(Duration::from_secs(1));
    endpoint.send(&ClientMessage::CtxIdle(pid)).unwrap();
    endpoint.send(&ClientMessage::CtxRelease(pid)).unwrap();
}
