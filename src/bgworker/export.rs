use crate::bgworker::rpc::Request;
use crate::bgworker::rpc::RequestBuild;
use crate::bgworker::rpc::RequestBuild2;
use crate::bgworker::rpc::RequestBulkdelete;
use crate::bgworker::rpc::RequestInsert;
use crate::bgworker::rpc::RequestSearch;
use crate::bgworker::rpc::ResponseBuild;
use crate::bgworker::rpc::ResponseBulkdelete;
use crate::bgworker::rpc::ResponseInsert;
use crate::bgworker::rpc::ResponseSearch;
use crate::bgworker::rpc::{read, write};
use crate::bgworker::Options;
use crate::postgres::HeapPointer;
use crate::postgres::Id;
use crate::postgres::Scalar;

pub fn build(
    id: Id,
    options: Options,
    mut parts: impl Iterator<Item = (Vec<Scalar>, HeapPointer)>,
) {
    let stream = std::net::TcpStream::connect("127.0.0.1:33509").unwrap();
    write(&stream, &Request::Build(RequestBuild { id, options }));
    while let Some((vector, p)) = parts.next() {
        write(&stream, &RequestBuild2::Continue { vector, p });
    }
    write(&stream, &RequestBuild2::Finish);
    let _res = read::<ResponseBuild>(&stream);
}

pub fn insert(id: Id, insert: (Vec<Scalar>, HeapPointer)) {
    let stream = std::net::TcpStream::connect("127.0.0.1:33509").unwrap();
    let req = Request::Insert(RequestInsert { id, insert });
    write(&stream, &req);
    let _res = read::<ResponseInsert>(&stream);
}

pub fn bulkdelete(id: Id, deletes: Vec<HeapPointer>) {
    let stream = std::net::TcpStream::connect("127.0.0.1:33509").unwrap();
    let req = Request::Bulkdelete(RequestBulkdelete { id, deletes });
    write(&stream, &req);
    let _res = read::<ResponseBulkdelete>(&stream);
}

pub fn search(id: Id, search: (Vec<Scalar>, usize)) -> Vec<HeapPointer> {
    let stream = std::net::TcpStream::connect("127.0.0.1:33509").unwrap();
    let req = Request::Search(RequestSearch { id, search });
    write(&stream, &req);
    let res = read::<ResponseSearch>(&stream);
    res.data
}
