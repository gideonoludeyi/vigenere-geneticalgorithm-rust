use std::{collections::HashMap, path::Path};

use clap::Parser;
use genevo::{algorithm::BestSolution, operator::prelude::*, prelude::*};

trait AsPhenotype {
    fn as_text(&self) -> &str;
}

type TextGenome = Vec<u8>;
impl AsPhenotype for TextGenome {
    fn as_text(&self) -> &str {
        std::str::from_utf8(self).unwrap()
    }
}

#[derive(Clone, Debug)]
struct Cipher(String);
impl Cipher {
    fn decrypt(&self, key: &TextGenome) -> Vec<u8> {
        let cipher: Vec<u8> = self
            .0
            .to_lowercase()
            .replace(|c: char| !c.is_alphabetic(), "")
            .into();
        let key: Vec<u8> = key
            .as_text()
            .to_lowercase()
            .replace(|c: char| !c.is_alphabetic(), "")
            .into();
        let mut plain = vec![];
        let mut key_ptr = 0_usize;
        for c in cipher {
            let mut key_char = 0;
            if !key.is_empty() {
                while key[key_ptr] < 97 || key[key_ptr] > (25 + 97) {
                    key_ptr = (key_ptr + 1) % key.len();
                }
                key_char = key[key_ptr];
                key_ptr = (key_ptr + 1) % key.len();
            }
            plain.push((26 + c - key_char) % 26 + 97);
        }
        plain
    }
}

#[derive(Clone, Debug)]
struct NGramCharFrequencyCalculator {
    cipher: Cipher,
    expected_freqdist: HashMap<String, f64>,
}
impl NGramCharFrequencyCalculator {
    fn new(cipher: Cipher, corpus: &str) -> Self {
        Self {
            cipher,
            expected_freqdist: Self::ngram_freq(corpus, 2),
        }
    }

    fn ngram_freq(text: &str, n: usize) -> HashMap<String, f64> {
        let length = text.len();
        let mut freq: HashMap<String, f64> = HashMap::new();
        for i in 0..=length - n {
            let s = &text[i..i + n];
            freq.insert(s.to_owned(), freq.get(s).unwrap_or(&0_f64) + 1.0);
        }
        for (_, count) in freq.iter_mut() {
            *count /= length as f64;
        }
        freq
    }

    fn cmpfreqdists(
        expected_freqdist: &HashMap<String, f64>,
        actual_freqdist: &HashMap<String, f64>,
    ) -> f64 {
        actual_freqdist
            .iter()
            .map(|(token, freq)| (freq - expected_freqdist.get(token).unwrap_or(&0.0)).abs())
            .sum()
    }
}
impl FitnessFunction<TextGenome, u32> for NGramCharFrequencyCalculator {
    fn fitness_of(&self, chromosome: &TextGenome) -> u32 {
        let plain = self.cipher.decrypt(chromosome);
        let freq = NGramCharFrequencyCalculator::ngram_freq(&String::from_utf8(plain).unwrap(), 2);
        let fit = NGramCharFrequencyCalculator::cmpfreqdists(&self.expected_freqdist, &freq);
        ((1.0 - fit) * 100.0) as u32
    }

    fn average(&self, values: &[u32]) -> u32 {
        values.iter().sum::<u32>() / values.len() as u32
    }

    fn highest_possible_fitness(&self) -> u32 {
        100
    }

    fn lowest_possible_fitness(&self) -> u32 {
        0
    }
}

struct DecryptionKeyGenomeBuilder {
    key_length: usize,
}
impl DecryptionKeyGenomeBuilder {
    fn new(key_length: usize) -> Self {
        Self { key_length }
    }
}
impl GenomeBuilder<Vec<u8>> for DecryptionKeyGenomeBuilder {
    fn build_genome<R>(&self, _: usize, rng: &mut R) -> TextGenome
    where
        R: Rng + Sized,
    {
        (0..self.key_length)
            .map(|_| {
                let value = rng.gen_range(97..=123);
                if value == 123 {
                    45 // "-"
                } else {
                    value
                }
            })
            .collect()
    }
}

#[derive(Debug)]
struct Params {
    initial_population_size: usize,
    chromosome_length: usize,
    max_generation_span: u64,
    crossover_rate: f64,
    mutation_rate: f64,
}

fn run(
    cipher: Cipher,
    params: &Params,
    corpuspath: &Path,
    seed: Option<Seed>,
) -> Option<BestSolution<TextGenome, u32>> {
    let population_builder = build_population()
        .with_genome_builder(DecryptionKeyGenomeBuilder::new(params.chromosome_length))
        .of_size(params.initial_population_size);
    let initial_population = match seed {
        Some(s) => population_builder.using_seed(s),
        None => population_builder.uniform_at_random(),
    };

    let fitness_function =
        NGramCharFrequencyCalculator::new(cipher, &std::fs::read_to_string(corpuspath).ok()?);
    let selection_op = TournamentSelector::new(params.crossover_rate, 2, 2, 1.0, false);
    let crossover_op = UniformCrossBreeder::new();
    let mutation_op = InsertOrderMutator::new(params.mutation_rate);
    let reinsertion_op = ElitistReinserter::new(
        fitness_function.clone(),
        true,
        1.0, /* no reinsertion */
    );
    let simulation_builder = simulate(
        genetic_algorithm()
            .with_evaluation(fitness_function)
            .with_selection(selection_op)
            .with_crossover(crossover_op)
            .with_mutation(mutation_op)
            .with_reinsertion(reinsertion_op)
            .with_initial_population(initial_population)
            .build(),
    )
    .until(GenerationLimit::new(params.max_generation_span));
    let mut simulation = match seed {
        Some(s) => simulation_builder.build_with_seed(s),
        None => simulation_builder.build(),
    };
    let mut best_solution = None;
    let mut bestfit = 0;
    loop {
        match simulation.step() {
            Ok(SimResult::Intermediate(step)) => {
                if step.result.best_solution.solution.fitness >= bestfit {
                    best_solution = Some(step.result.best_solution.clone());
                    bestfit = step.result.best_solution.solution.fitness;
                }
            }
            Ok(SimResult::Final(step, _processing_time, _duration, _stop_reason)) => {
                if step.result.best_solution.solution.fitness >= bestfit {
                    best_solution = Some(step.result.best_solution.clone());
                }
                break;
            }
            Err(error) => {
                println!("{}", error);
                break;
            }
        }
    }
    best_solution
}

mod cli_decrypt {
    use super::{AsPhenotype, Cipher};

    #[derive(clap::Args)]
    pub struct Args {
        #[arg(short, long)]
        key: String,

        #[arg(short, long, default_value = "tmp/cipher.txt")]
        inputfile: std::path::PathBuf,
    }

    pub fn execute(args: Args) {
        let ciphertext = std::fs::read_to_string(&args.inputfile).unwrap();
        let cipher = Cipher(ciphertext);
        let plaintext = cipher.decrypt(&args.key.as_bytes().to_vec());
        println!("{}", plaintext.as_text());
    }
}

mod cli_run {
    use super::{run, AsPhenotype, Cipher, Params};
    use genevo::prelude::*;

    #[derive(clap::Args)]
    pub struct Args {
        #[arg(short, long, default_value = "tmp/cipher.txt")]
        inputfile: std::path::PathBuf,

        #[arg(long, default_value = "tmp/rawtext.txt")]
        corpusfile: std::path::PathBuf,

        #[arg(short, long, default_value_t = 1000)]
        population: usize,

        #[arg(short, long, default_value_t = 8)]
        keylength: usize,

        #[arg(short, long, default_value_t = 50)]
        generations: u64,

        #[arg(short, long, default_value_t = 0.9)]
        crossover_rate: f64,

        #[arg(short, long, default_value_t = 0.1)]
        mutation_rate: f64,

        #[arg(short, long)]
        seed: Option<String>,
    }

    pub fn execute(args: Args) {
        let ciphertext = std::fs::read_to_string(&args.inputfile).unwrap();
        let cipher = Cipher(ciphertext);
        let params = Params {
            initial_population_size: args.population,
            chromosome_length: args.keylength,
            max_generation_span: args.generations,
            crossover_rate: args.crossover_rate,
            mutation_rate: args.mutation_rate,
        };
        let seed: Option<Seed> = args.seed.and_then(|seed| {
            seed.bytes()
                .cycle()
                .take(32) // seed has to be 32-bytes
                .collect::<Vec<u8>>()
                .try_into()
                .ok()
            // Note: will be None if seed is an empty string
        });
        let corpuspath = std::path::Path::new(&args.corpusfile);
        if let Some(best) = run(cipher, &params, corpuspath, seed) {
            println!(
                "Final result: best solution with fitness {} found in generation {}",
                best.solution.fitness, best.generation,
            );
            println!("{}", best.solution.genome.as_text());
        } else {
            eprintln!("Something went wrong");
        }
    }
}

#[derive(clap::Parser)]
#[command(version, about, long_about = None, propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    Run(cli_run::Args),
    Decrypt(cli_decrypt::Args),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cli_run::execute(args),
        Commands::Decrypt(args) => cli_decrypt::execute(args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CIPHER_TEXT: &str = "xbwdesmhihslwhkktefvktkktcwfpiibihwmosfilojvooegvefwnochsuuspsureifakbnlalzsrsroiejwzgfpjczldokrceoahzshpbdwpcjstacgbarfwifwohylckafckzwwomlalghrtafchfetcgfpfrgxclwzocdctmjebx";
    const PLAIN_TEXT: &str = "ibelievethatattheendofthecenturytheuseofwordsandgeneraleducatedopinionwillhavealteredsomuchthatonewillbeabletospeakofmachinesthinkingwithoutexpectingtobecontradictedalanturing";

    #[test]
    fn test_decrypt() {
        let plain = Cipher(CIPHER_TEXT.to_owned()).decrypt(&b"password".into());
        let actual = String::from_utf8(plain).unwrap();
        let expected = PLAIN_TEXT.to_owned();
        assert!(
            expected == actual,
            "expected `{}`, got `{}`",
            &expected,
            &actual
        );
    }
}
