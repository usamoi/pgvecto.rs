use crate::ipc::ClientIpcError;
use crate::ipc::ServerIpcError;
use byteorder::{ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::net::TcpStream;
use std::net::ToSocketAddrs;

macro_rules! resolve_server_closed {
    ($t: expr) => {
        match $t {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(ServerIpcError::Closed)
            }
            Err(e) => panic!("{}", e),
            Ok(e) => e,
        }
    };
}

macro_rules! resolve_client_closed {
    ($t: expr) => {
        match $t {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(ClientIpcError::Closed)
            }
            Err(e) => panic!("{}", e),
            Ok(e) => e,
        }
    };
}

pub struct Listener {
    listener: TcpListener,
}

impl Listener {
    pub fn new(path: impl ToSocketAddrs) -> Self {
        let listener = TcpListener::bind(&path).expect("Failed to bind.");
        Self { listener }
    }
    pub fn accept(&mut self) -> Socket {
        let (stream, _) = self.listener.accept().expect("Failed to listen.");
        Socket {
            stream: Some(stream),
        }
    }
}

pub struct Socket {
    stream: Option<TcpStream>,
}

impl Socket {
    pub fn new(path: impl ToSocketAddrs) -> Self {
        let stream = TcpStream::connect(path).expect("Failed to bind.");
        Socket {
            stream: Some(stream),
        }
    }
    pub fn server_send<T>(&mut self, packet: T) -> Result<(), ServerIpcError>
    where
        T: Serialize,
    {
        use byteorder::NativeEndian as N;
        let stream = self.stream.as_mut().ok_or(ServerIpcError::Closed)?;
        let buffer = bincode::serialize(&packet).expect("Failed to serialize");
        let len = u32::try_from(buffer.len()).expect("Packet is too large.");
        resolve_server_closed!(stream.write_u32::<N>(len));
        resolve_server_closed!(stream.write_all(&buffer));
        Ok(())
    }
    pub fn client_recv<T>(&mut self) -> Result<T, ClientIpcError>
    where
        T: for<'a> Deserialize<'a>,
    {
        use byteorder::NativeEndian as N;
        let stream = self.stream.as_mut().ok_or(ClientIpcError::Closed)?;
        let len = resolve_client_closed!(stream.read_u32::<N>());
        let mut buffer = vec![0u8; len as usize];
        resolve_client_closed!(stream.read_exact(&mut buffer));
        let packet = bincode::deserialize(&buffer).expect("Failed to deserialize.");
        Ok(packet)
    }
    pub fn client_send<T>(&mut self, packet: T) -> Result<(), ClientIpcError>
    where
        T: Serialize,
    {
        use byteorder::NativeEndian as N;
        let stream = self.stream.as_mut().ok_or(ClientIpcError::Closed)?;
        let buffer = bincode::serialize(&packet).expect("Failed to serialize");
        let len = u32::try_from(buffer.len()).expect("Packet is too large.");
        resolve_client_closed!(stream.write_u32::<N>(len));
        resolve_client_closed!(stream.write_all(&buffer));
        Ok(())
    }
    pub fn server_recv<T>(&mut self) -> Result<T, ServerIpcError>
    where
        T: for<'a> Deserialize<'a>,
    {
        use byteorder::NativeEndian as N;
        let stream = self.stream.as_mut().ok_or(ServerIpcError::Closed)?;
        let len = resolve_server_closed!(stream.read_u32::<N>());
        let mut buffer = vec![0u8; len as usize];
        resolve_server_closed!(stream.read_exact(&mut buffer));
        let packet = bincode::deserialize(&buffer).expect("Failed to deserialize.");
        Ok(packet)
    }
}
