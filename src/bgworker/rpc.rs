use super::{global_index_build, global_index_load, Options};
use crate::postgres::{HeapPointer, Id, Scalar};
use byteorder::ReadBytesExt;
use byteorder::{WriteBytesExt, LE};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::io::Read;
use std::io::Write;
use std::net::TcpStream;

pub fn write<T: Serialize>(mut stream: impl Write, value: &T) {
    let bytes = bincode::serialize(value).unwrap();
    assert!(bytes.len() <= u16::MAX as usize);
    stream.write_u16::<LE>(bytes.len() as u16).unwrap();
    stream.write_all(&bytes).unwrap();
}

pub fn read<T: DeserializeOwned>(mut stream: impl Read) -> T {
    let len = stream.read_u16::<LE>().unwrap();
    let mut buffer = vec![0u8; len as usize];
    stream.read_exact(&mut buffer).unwrap();
    bincode::deserialize(&buffer).unwrap()
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum Request {
    Build(RequestBuild),
    Insert(RequestInsert),
    Bulkdelete(RequestBulkdelete),
    Search(RequestSearch),
}

pub fn thread_session(stream: TcpStream) -> ! {
    use Request::*;
    loop {
        let request = read::<Request>(&stream);
        match request {
            Build(req) => {
                struct Iter<'a> {
                    stream: &'a TcpStream,
                    finished: bool,
                }
                impl<'a> Iterator for Iter<'a> {
                    type Item = (Vec<Scalar>, HeapPointer);

                    fn next(&mut self) -> Option<Self::Item> {
                        use RequestBuild2::*;
                        if self.finished {
                            return None;
                        }
                        if let Continue { vector, p } = read::<RequestBuild2>(self.stream) {
                            Some((vector, p))
                        } else {
                            self.finished = true;
                            None
                        }
                    }
                }
                build(
                    req.id,
                    req.options,
                    Iter {
                        stream: &stream,
                        finished: false,
                    },
                );
                let res = ResponseBuild {};
                write(&stream, &res);
            }
            Insert(req) => {
                insert(req.id, req.insert);
                let res = ResponseInsert {};
                write(&stream, &res);
            }
            Bulkdelete(req) => {
                bulkdelete(req.id, req.deletes);
                let res = ResponseBulkdelete {};
                write(&stream, &res);
            }
            Search(req) => {
                let data = search(req.id, req.search);
                let res = ResponseSearch { data };
                write(&stream, &res);
            }
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RequestBuild {
    pub id: Id,
    pub options: Options,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum RequestBuild2 {
    Continue { vector: Vec<Scalar>, p: HeapPointer },
    Finish,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ResponseBuild {}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RequestInsert {
    pub id: Id,
    pub insert: (Vec<Scalar>, HeapPointer),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ResponseInsert {}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RequestBulkdelete {
    pub id: Id,
    pub deletes: Vec<HeapPointer>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ResponseBulkdelete {}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct RequestSearch {
    pub id: Id,
    pub search: (Vec<Scalar>, usize),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ResponseSearch {
    pub data: Vec<HeapPointer>,
}

fn build(id: Id, options: Options, parts: impl Iterator<Item = (Vec<Scalar>, HeapPointer)>) {
    global_index_build(id, options, parts);
}

fn insert(id: Id, insert: (Vec<Scalar>, HeapPointer)) {
    let index = global_index_load(id);
    let mut index = index.write().unwrap();
    index.insert(insert);
}

fn bulkdelete(id: Id, deletes: Vec<HeapPointer>) {
    let index = global_index_load(id);
    let mut index = index.write().unwrap();
    index.bulkdelete(&deletes);
}

fn search(id: Id, search: (Vec<Scalar>, usize)) -> Vec<HeapPointer> {
    let index = global_index_load(id);
    let mut index = index.write().unwrap();
    index.search(search)
}
