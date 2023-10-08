use crate::ipc::ClientIpcError;
use crate::ipc::ServerIpcError;
use serde::Deserialize;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum Address {
    Tcp(std::net::SocketAddr),
    Unix(PathBuf),
}

pub enum Listener {
    Tcp(super::transport_tcp::Listener),
    Unix(super::transport_unix::Listener),
}

impl Listener {
    pub fn new(addr: Address) -> Self {
        match addr {
            Address::Tcp(x) => Listener::Tcp(super::transport_tcp::Listener::new(x)),
            Address::Unix(x) => Listener::Unix(super::transport_unix::Listener::new(x)),
        }
    }
    pub fn accept(&mut self) -> Socket {
        match self {
            Listener::Tcp(x) => Socket::Tcp(x.accept()),
            Listener::Unix(x) => Socket::Unix(x.accept()),
        }
    }
}

pub enum Socket {
    Tcp(super::transport_tcp::Socket),
    Unix(super::transport_unix::Socket),
}

impl Socket {
    pub fn new(addr: Address) -> Self {
        match addr {
            Address::Tcp(x) => Socket::Tcp(super::transport_tcp::Socket::new(x)),
            Address::Unix(x) => Socket::Unix(super::transport_unix::Socket::new(x)),
        }
    }
    pub fn server_send<T>(&mut self, packet: T) -> Result<(), ServerIpcError>
    where
        T: Serialize,
    {
        match self {
            Socket::Tcp(x) => x.server_send(packet),
            Socket::Unix(x) => x.server_send(packet),
        }
    }
    pub fn client_recv<T>(&mut self) -> Result<T, ClientIpcError>
    where
        T: for<'a> Deserialize<'a>,
    {
        match self {
            Socket::Tcp(x) => x.client_recv(),
            Socket::Unix(x) => x.client_recv(),
        }
    }
    pub fn client_send<T>(&mut self, packet: T) -> Result<(), ClientIpcError>
    where
        T: Serialize,
    {
        match self {
            Socket::Tcp(x) => x.client_send(packet),
            Socket::Unix(x) => x.client_send(packet),
        }
    }
    pub fn server_recv<T>(&mut self) -> Result<T, ServerIpcError>
    where
        T: for<'a> Deserialize<'a>,
    {
        match self {
            Socket::Tcp(x) => x.server_recv(),
            Socket::Unix(x) => x.server_recv(),
        }
    }
}
