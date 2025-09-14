use std::error::Error;
use std::fs::File;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::env::Args;

use redis::{Client};
use once_cell::sync::Lazy;

const DEV: bool = true;
const DEFAULT_DATASET: &str = "embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt";

static REDIS_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::open("redis://localhost:6379").expect("Failed to establish redis client")
});


fn main() -> Result<(), Box<dyn Error>>{
    let embeddings_hashmap = init_hashmap();

    let mut conn = REDIS_CLIENT.get_connection()?;

    let pong: String = redis::cmd("PING").query(&mut conn)?;
    println!("Redis response to ping: {}", pong);

    let word1 = "king";
    let word2 = "queen";
    let word1_emb = embeddings_hashmap.get(word1).expect("word not found in the dataset");
    let word2_emb = embeddings_hashmap.get(word2).expect("word not found in the dataset");
    let similarity = similarity(word1_emb, word2_emb);
    println!("Similarity between '{word1}' and '{word2}' is: {similarity}");

    Ok(())
}

fn similarity(word1: &[f32], word2: &[f32]) -> f32 {
    let dot_product: f32 = word1.iter().zip(word2).map(|(a, b)| a * b).sum();
    let norm1: f32 = word1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = word2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm1 * norm2)
}

fn preprocessing(dataset: String) -> HashMap<String, Vec<f32>> {
    let mut embeddings_hashmap: HashMap<String, Vec<f32>> = HashMap::with_capacity(1_300_000);

    let file = File::open(dataset).unwrap();
    let embedding_buffer = BufReader::new(file);

    embedding_buffer.lines().for_each(|line| {
        let line = line.unwrap();
        let mut iter = line.split_whitespace();
        let word = iter.next().unwrap().to_string();
        let embedding: Vec<f32> = iter.map(|x| x.parse().expect("invalid input")).collect();
        embeddings_hashmap.insert(word, embedding);
    });

    embeddings_hashmap
}

fn init_hashmap() -> HashMap<String, Vec<f32>> {
    if DEV {
        preprocessing(String::from(DEFAULT_DATASET))
    } else {
        let mut args: Args = std::env::args();
        let dataset = args.nth(1).unwrap();
        preprocessing(dataset)
    }
}