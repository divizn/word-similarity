use log::info;
use tokio::io::{self, AsyncBufReadExt, BufReader};
use tokio::fs::File;
use redis::{AsyncCommands, Client};
use std::error::Error;
use std::sync::LazyLock;

const DEV: bool = true;
const BATCH_SIZE: usize = 500; // Number of embeddings per pipeline batch for Redis HSET

static REDIS_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::open("redis://localhost:6379").expect("Failed to establish redis client")
});

/// Represents the input word. Wraps the input word with Embedding  so it can be retreived from Redis.
struct Token {
    word: String, // e.g. king
    embedding: Embedding, // e.g. see Embedding
}

struct Word {
    val: String, // word value e.g. king
    vector: Option<Vec<f32>> // vector representation

}

/// Represents the dataset, and wraps with Embedding so can provide context to Redis.
struct Dataset {
    dir: String, // e.g. embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt
    embedding: Embedding // see Embedding
}

/// The different types of embeddings stored in Redis.
enum Embedding {
    Glove,
    Word2vec,
}

impl Embedding {
    /// Gets the Embedding from the directory path.
    fn from_path(path: &str) -> Option<Self> {
        if path.to_uppercase().contains("GLOVE") {
            Some(Embedding::Glove)
        } else if path.to_uppercase().contains("WORD2VEC") {
            Some(Embedding::Word2vec)
        } else {
            None
        }
    }

    /// `as_str` implementation to get embedding from Embedding.
    fn as_str(&self) -> &str {
        match self {
            Embedding::Glove => "GLOVE",
            Embedding::Word2vec => "WORD2VEC",
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    info!("Logger initialised");

    let mut conn = REDIS_CLIENT.get_multiplexed_async_connection().await?;

    let db_ready: bool = conn.exists("GLOVE:OK").await?;
    if db_ready {
        info!("Embeddings already in Redis, skipping streaming.");
    } else {
        info!("Streaming embeddings into Redis...");
        init_db(&mut conn).await?;
        info!("Finished streaming embeddings.");
    }

    let mut word1 = Word{val: String::from("dog"), vector: None};
    let mut word2 = Word{val: String::from("cat"), vector: None};

    word1.vector = Some(fetch_embedding(&mut conn, Token{word:word1.val.clone(), embedding:Embedding::Glove}).await?);
    word2.vector = Some(fetch_embedding(&mut conn, Token{word:word2.val.clone(), embedding:Embedding::Glove}).await?);

    let similarity = cosine_similarity(&word1.vector.unwrap(), &word2.vector.expect("vector not found for vector"));
    println!("Similarity between '{}' and '{}': {similarity}", word1.val, word2.val);

    Ok(())
}

/// Takes Multiplexed Redis connection and Token and fetches the vector from Redis provided by `Token.word`.
async fn fetch_embedding(
    conn: &mut redis::aio::MultiplexedConnection,
    token: Token,
) -> redis::RedisResult<Vec<f32>> {
    let redis_key = format!("{}:{}", token.embedding.as_str(), token.word);
    let result: Vec<String> = conn.hvals(redis_key).await?;
    Ok(result
        .into_iter()
        .map(|v| v.parse::<f32>().unwrap())
        .collect())
}

/// Calculates the cosine simuilarity between 2 vectors `vec1` and `vec2`.
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot: f32 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();
    let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm1 * norm2)
}

/// Sets up Redis to store vectors provided by the `Dataset`. The key is represented like `Embedding:word` where `dataset.Embedding` is the embedding, and word is the word from the dataset.
async fn load_db(
    dataset: Dataset,
    conn: &mut redis::aio::MultiplexedConnection,
) -> io::Result<()> {
    let file = File::open(&dataset.dir).await?;
    info!("opened file {}", &dataset.dir);
    let reader = BufReader::new(file);
    info!("reading lines from file");
    let mut lines = reader.lines();

    let mut pipe = redis::pipe();
    let mut batch_count = 0;
    let mut total_count = 0;

    info!("starting redis storage");

    while let Some(line) = lines.next_line().await? {
        let mut iter = line.split_whitespace();
        if let Some(word) = iter.next() {
            let embedding: Vec<f32> = iter.map(|x| x.parse().unwrap()).collect();
            let redis_key = format!("{}:{}", dataset.embedding.as_str(), word);
            info!("word: {word}");
            info!("embedding (first only): {:?}", embedding[0]);
            info!("created redis key for {word}: {redis_key}");

            info!("piping HSETs");
            pipe.cmd("HSET")
                .arg(&redis_key)
                .arg(
                    embedding
                        .iter()
                        .enumerate()
                        .flat_map(|(i, val)| vec![i.to_string(), val.to_string()])
                        .collect::<Vec<String>>(),
                );
            
            batch_count += 1;
            total_count += 1;
            info!("current batch size: {batch_count}");
            info!("total batch size: {total_count}");

            if batch_count >= BATCH_SIZE {
                let _: () = pipe.query_async(conn).await.unwrap();
                pipe = redis::pipe(); // reset pipeline
                batch_count = 0;
                info!("Added {batch_count} embeddings to Redis (total {total_count})");
            }
        }
    }

    // Flush remaining commands
    if batch_count > 0 {
        let _: () = pipe.query_async(conn).await.unwrap();
        info!("Added final {batch_count} embeddings to Redis (total {total_count})");
    }
    
    let ok_key = format!("{}:OK", dataset.embedding.as_str());
    let _: () = redis::cmd("SET")
        .arg(&ok_key)
        .arg("1")
        .query_async(conn)
        .await
        .unwrap();

    info!("Finished streaming embeddings to Redis {ok_key}");

    Ok(())
}

/// Retrieves dataset from stdin, else fallbacks to default directory (glove.twitter.27B).
async fn init_db(conn: &mut redis::aio::MultiplexedConnection) -> io::Result<()> {
    let dataset = if DEV {
        Dataset{
            dir: String::from("embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt"),
            embedding: Embedding::from_path("embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt").expect("Unknown embedding type")
        }
    } else {
        let arg = std::env::args().nth(1).expect("No dataset argument provided");
        Dataset{
            dir: String::from(&arg),
            embedding: Embedding::from_path(arg.as_str()).expect("Unknown embedding type")
        }
    };

    load_db(dataset, conn).await
}