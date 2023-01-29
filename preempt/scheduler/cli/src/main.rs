use clap::{Parser, Subcommand};
use transport::{
    msg::{ClientMessage, ServerMessage},
    socket::ClientEndpoint,
};

#[derive(Subcommand)]
enum Commands {
    /// Get priority of context
    #[command(name = "get")]
    GetPriorityCmd {
        /// The pid of the context
        #[arg(short, long)]
        pid: u32,
    },
    /// Set priority of context
    #[command(name = "set")]
    SetPriorityCmd {
        /// The pid of the context
        #[arg(short, long)]
        pid: u32,
        /// The priority of the context
        #[arg(short = 'v', long)]
        priority: i32,
    },
}

#[derive(Parser)]
struct Opts {
    #[command(subcommand)]
    cmd: Commands,
}

fn main() {
    let opts = Opts::parse();
    let endpoint = ClientEndpoint::new_connected().unwrap();
    match opts.cmd {
        Commands::GetPriorityCmd { pid } => {
            eprintln!("get priority of context {}", pid);
            let msg = ClientMessage::GetCtxPriority(pid);
            endpoint.send(&msg).unwrap();
            if let ServerMessage::CtxPriority { priority, .. } = endpoint.recv().unwrap() {
                println!("{priority}");
            } else {
                std::process::exit(1);
            }
        }
        Commands::SetPriorityCmd { pid, priority } => {
            eprintln!("set priority of context {} to {}", pid, priority);
            let msg = ClientMessage::SetCtxPriority { pid, priority };
            endpoint.send(&msg).unwrap();
        }
    }
}
