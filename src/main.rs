use std::fs::File;
use std::collections::HashMap;
use std::io::Read;
use std::env::Args;

const DEV: bool = true;
const DEFAULT_DATASET: &str = "datasets/glove.twitter.27B/glove.twitter.27B.200d.txt";

fn main() {
    let embeddings_hashmap = init_hashmap();

    let word1 = "king";
    let word2 = "queen";
    let word1_emb = embeddings_hashmap.get(word1).expect("word not found in the dataset");
    let word2_emb = embeddings_hashmap.get(word2).expect("word not found in the dataset");
    let similarity = similarity(word1_emb, word2_emb);
    println!("Similarity between '{word1}' and '{word2}' is: {similarity}");
}

fn similarity(word1: &[f32], word2: &[f32]) -> f32 {
    let mut dot_product: f32 = 0.0;
    let mut norm_word1: f32 = 0.0;
    let mut norm_word2: f32 = 0.0;

    for i in 0..word1.len() {
        dot_product += word1[i] * word2[i];
        norm_word1 += word1[i] * word1[i];
        norm_word2 += word2[i] * word2[i];
    }

    dot_product / (norm_word1.sqrt() * norm_word2.sqrt())
}

fn preprocessing(dataset: String) -> HashMap<String, Vec<f32>> {
    let mut embeddings_hashmap: HashMap<String, Vec<f32>> = HashMap::new();

    let mut reader = File::open(dataset).unwrap();
    let mut embedding_buffer: String = String::new();
    match reader.read_to_string(&mut embedding_buffer) {
        Ok(bytes) => println!("Read {bytes} bytes ({mb}MB)", mb = bytes / 1024 / 1024),
        Err(e) => {println!("{e}")},
    }
    embedding_buffer.lines().for_each(|line| {
        let mut iter = line.split_whitespace();
        let word = iter.next().unwrap().to_string();
        let embedding: Vec<f32> = iter.map(|x| x.parse().expect("invalid input")).collect();
        embeddings_hashmap.insert(word, embedding);
    });

    embeddings_hashmap
}

fn init_hashmap() -> HashMap<String, Vec<f32>> {
    let default = String::from(DEFAULT_DATASET);
    if DEV {
        preprocessing(default)
    } else {
        let mut args: Args = std::env::args();
        let dataset = args.nth(1).unwrap();
        preprocessing(dataset)
    }
}