use clap::Parser;
use service::{bgworker::WorkerOptions, ipc::transport::Address};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    addr: SocketAddr,
    #[arg(long)]
    chdir: PathBuf,
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.chdir).expect("Failed to create the directory.");
    service::bgworker::main(WorkerOptions {
        addr: Address::Tcp(args.addr),
        chdir: args.chdir,
    });
}
