use std::mem::size_of;

use bytecheck::CheckBytes;
use rkyv::{
    ser::{serializers::BufferSerializer, Serializer},
    AlignedBytes, Archive, Archived, Deserialize, Infallible, Serialize,
};

/// Messages send from client to server
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Archive)]
#[archive_attr(derive(CheckBytes))]
pub enum ClientMessage {
    CtxCreated(u32),
    /// The context has unfinished tasks
    CtxReady(u32),
    /// All tasks of the context are finished
    CtxIdle(u32),
    CtxRelease(u32),
    SetCtxPriority { pid: u32, priority: i32 },
    GetCtxPriority(u32),
}

impl ClientMessage {
    pub const SIZE: usize = size_of::<Archived<Self>>();

    pub(crate) fn from_bytes(buf: &AlignedBytes<{ Self::SIZE }>) -> Self {
        let archived = rkyv::check_archived_value::<Self>(buf.as_slice(), 0).unwrap();
        archived.deserialize(&mut Infallible).unwrap()
    }

    pub(crate) fn to_bytes(&self, buf: &mut [u8]) {
        let mut serializaer = BufferSerializer::new(buf);
        serializaer.serialize_value(self).unwrap();
    }
}

/// Messages send from server to client
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Archive)]
#[archive_attr(derive(CheckBytes))]
pub enum ServerMessage {
    StopCtx,
    ResumeCtx,
    CtxPriority { pid: u32, priority: i32 },
}

impl ServerMessage {
    pub const SIZE: usize = size_of::<Archived<Self>>();

    pub(crate) fn from_bytes(buf: &AlignedBytes<{ Self::SIZE }>) -> Self {
        let archived = rkyv::check_archived_value::<Self>(buf.as_slice(), 0).unwrap();
        archived.deserialize(&mut Infallible).unwrap()
    }

    pub(crate) fn to_bytes(&self, buf: &mut [u8]) {
        let mut serializaer = BufferSerializer::new(buf);
        serializaer.serialize_value(self).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_message() {
        let msg = ClientMessage::CtxIdle(0xffeeffee);
        let mut buf = AlignedBytes::default();
        msg.to_bytes(buf.as_mut_slice());
        //println!("{:?}", buf);
        let msg2 = ClientMessage::from_bytes(&buf);
        assert_eq!(msg, msg2);
    }

    #[test]
    fn test_server_message() {
        let msg = ServerMessage::CtxPriority {
            pid: 0xffeeffee,
            priority: 2,
        };
        let mut buf = AlignedBytes::default();
        msg.to_bytes(buf.as_mut_slice());
        let msg2 = ServerMessage::from_bytes(&buf);
        assert_eq!(msg, msg2);
    }
}
