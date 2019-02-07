pub mod examples;

use std::io;
use std::io::prelude::*;
use std::process::Command;

fn main() {
    loop {
        println!("Machine Learning Examples: ");
        println!("  0) Quit");
        println!("  1) Learning XOR Gate with Neural Network");
        
        let mut stdout = io::stdout();
        write!(stdout, "\nEnter option number: ").unwrap();
        stdout.flush().unwrap();
        
        let mut option = String::new();
        io::stdin().read_line(&mut option).unwrap();
        let option: i32 = option.trim().parse().unwrap_or(-1);

        println!();

        match option {
            0 => break,
            1 => examples::neural_network::learing_xor_gate(),
            _ => {
                println!("Invalid option!\n");
                continue;
            }
        }
        
        if cfg!(windows) {
            println!();
            let _ = Command::new("cmd.exe").arg("/c").arg("pause").status();
        } else {
            write!(stdout, "\nPress any any key to continue...").unwrap();
            stdout.flush().unwrap();
            let _ = io::stdin().read(&mut [0u8]).unwrap();
        }

        println!();
    }
}