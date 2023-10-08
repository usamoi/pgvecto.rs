pub mod client;
mod packet;
pub mod server;
pub mod transport;
mod transport_tcp;
mod transport_unix;

use self::server::RpcHandler;
use self::{client::Rpc, transport::Address};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum ServerIpcError {
    #[error("The connection is closed.")]
    Closed,
    #[error("Server encounters an error.")]
    Server,
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ClientIpcError {
    #[error("The connection is closed.")]
    Closed,
    #[error("Server encounters an error.")]
    Server,
}

pub fn listen(addr: Address) -> impl Iterator<Item = RpcHandler> {
    let mut listener = self::transport::Listener::new(addr);
    std::iter::from_fn(move || {
        let socket = listener.accept();
        Some(self::server::RpcHandler::new(socket))
    })
}

pub fn connect(addr: Address) -> Rpc {
    let socket = self::transport::Socket::new(addr);
    self::client::Rpc::new(socket)
}
