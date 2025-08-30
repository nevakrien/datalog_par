use datalog_par::parser::KB;
use datalog_par::run_many_or_exit;
use std::env;


fn usage(exe: &str) {
    eprintln!(
        "usage: {exe} [FILE ...]
Load and execute Datalog files sequentially using a shared knowledge base.

Options:
  -h, --help    Show this help message"
    );
}

fn main() {
    let exe = env::args().next().unwrap_or_else(|| "datalog".into());
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage(&exe);
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }

    // Single KB for the whole run; swap to KB::new() if that's your constructor.
    let mut kb = KB::default();
    run_many_or_exit(&mut kb, &args);
}
