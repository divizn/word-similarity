use std::error::Error;
use std::collections::HashMap;
use std::env::Args;

use tokio::io::{self, AsyncBufReadExt, BufReader};
use tokio::fs::File;
use tokio::task;

use redis::{Client};
use once_cell::sync::Lazy;

const DEV: bool = true;
const DEFAULT_DATASET: &str = "embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt";

static REDIS_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::open("redis://localhost:6379").expect("Failed to establish redis client")
});

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>>{
    let embeddings_handle = task::spawn(async {
        init_hashmap().await
    });


    let mut conn = REDIS_CLIENT
    .get_multiplexed_async_connection().await?;

    let pong: String = redis::cmd("PING").query_async(&mut conn).await?;
    println!("Redis response to ping: {}", pong);
    
    let embeddings_hashmap = embeddings_handle.await??;


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

async fn preprocessing(dataset: String) -> io::Result<HashMap<String, Vec<f32>>> {
    let mut embeddings_hashmap: HashMap<String, Vec<f32>> = HashMap::with_capacity(1_300_000);

    let file = File::open(dataset).await?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await? {
        let mut iter = line.split_whitespace();
        if let Some(word) = iter.next() {
            let embedding: Vec<f32> = iter.map(|x| x.parse().expect("invalid input")).collect();
            embeddings_hashmap.insert(word.to_string(), embedding);
        }
    }

    Ok(embeddings_hashmap)
}

async fn init_hashmap() -> io::Result<HashMap<String, Vec<f32>>> {
    let dataset = if DEV {
        String::from(DEFAULT_DATASET)
    } else {
        let mut args: Args = std::env::args();
        args.nth(1).expect("No dataset argument provided")
    };

    preprocessing(dataset).await
}