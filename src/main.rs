use axum::http::StatusCode;
use axum::routing::{post};
use axum::{Json, Router};
use log::{error, info, warn};
use redis::aio::MultiplexedConnection;
use serde::{Deserialize, Serialize};
use tokio::io::{self, AsyncBufReadExt, BufReader};
use tokio::fs::File;
use redis::{AsyncCommands, Client, RedisError};
use tokio::net::TcpListener;
use tokio::time::{Duration};
use std::error::Error;
use std::sync::{LazyLock};

/// DEV variable for dataseet
const DEV: bool = true;
/// Number of embeddings per pipeline batch for Redis HSET
const BATCH_SIZE: usize = 500; 
/// Max amount of retries before stopping to attempt Redis query/connection
const REDIS_MAX_RETRY_COUNT: u8 = 5;
/// Duration to wait between each retry
const REDIS_RETRY_DELAY_MS: u64 = 2000;

static REDIS_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::open("redis://localhost:6379").expect("Failed to establish redis client")
});


/// Represents the input word. Wraps the input word with Embedding  so it can be retreived from Redis.
struct Token {
    word: String, // e.g. king
    embedding: Embedding, // e.g. see Embedding
}

/// Represents a word, stores word value and the vector representation - different from Token which is used for loading words into database
#[derive(Serialize, Deserialize)]
struct Word {
    val: String, // word value e.g. king
    vector: Option<Vec<f32>> // vector representation

}

/// Represents the dataset, and wraps with Embedding so can provide context to Redis.
struct Dataset {
    dir: String, // e.g. embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt
    embedding: Embedding // see Embedding


}


/// ### API Request body
/// - `word1` - First word to compare
/// - `word2` - Second word to compare
/// - `embedding` - The vector embedding to use e.g. `GLOVE` or `WORD2VEC`. Defaults to `GLOVE`
#[derive(Deserialize)]
struct SimilarityRequest {
    word1: Word,
    word2: Word,

    #[serde(default = "default_embedding")]
    embedding: Embedding, // "Glove" or "Word2vec   "

    #[serde(default = "default_measure")]
    measure: Measure
}

/// Returns the default `Embedding`
fn default_embedding() -> Embedding {
    Embedding::Glove
}

// Returns the default `Measure`.
fn default_measure() -> Measure {
    Measure::Cosine
}

/// ### API Response body
/// - `similarity` - returns the similarity of the 2 words.
/// - `measure` - returns the `measure` used e.g. cosine similarity
#[derive(Serialize)]
struct SimilarityResponse {
    words: [Word; 2],
    similarity: f32,
}

/// Measures used to estimate similarity between words.
#[derive(Serialize, Deserialize)]
enum Measure {
    Cosine,
    Dot
}

/// The different types of embeddings stored in Redis.
#[derive(Deserialize, Clone, Copy)]
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
    /// `as_str` implementation to get embedding str from Embedding.
    fn as_str(&self) -> &str {
        match self {
            Embedding::Glove => "GLOVE",
            Embedding::Word2vec => "WORD2VEC",
        }
    }
}

async fn similarity_handler(
    Json(payload): Json<SimilarityRequest>,
) -> Result<Json<SimilarityResponse>, StatusCode> {
    // Fetch embeddings from Redis
    let token1 = Token {
        word: payload.word1.val.clone(),
        embedding: payload.embedding,
    };

    let token2 = Token {
        word: payload.word2.val.clone(),
        embedding: payload.embedding,
    };

    let mut conn1 = REDIS_CLIENT
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| {
            if e.is_connection_refusal() {
                error!("Redis connection refused: {e}");
            } else {
                error!("Redis error: {e}");
            }
            StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // TODO: fix this so it works
    //     let mut conn1 = get_redis_conn_with_retries()
    //     .await
    //     .map_err(|e| {
    //         if e.kind() == redis::ErrorKind::BusyLoadingError {
    //             error!("Redis could not load after 5 connections in handler, returning status 500");
    //         } else {
    //             error!("Unknown Redis error occured in handler attempting to get connection: {e}");     
    //         }
    //         StatusCode::INTERNAL_SERVER_ERROR
    // })?;

    let mut conn2 = REDIS_CLIENT
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| {
            if e.is_connection_refusal() {
                error!("Redis connection refused: {e}")
            } else {
                error!("Redis error: {e}");
            }
            StatusCode::INTERNAL_SERVER_ERROR
    })?;

    
    let vec1 = fetch_embedding(&mut conn1, token1)
        .await
        .map_err(|e|{
            println!("{e}, kind {:?}", e.kind());
            if e.kind() == redis::ErrorKind::BusyLoadingError {
                error!("Redis is not loaded yet causing 500 error: {e}")
            } else {
                error!("Redis error: {e}");
            }
            StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let vec2 = fetch_embedding(&mut conn2, token2)    
        .await
        .map_err(|e|{
            if e.kind() == redis::ErrorKind::BusyLoadingError {
                warn!("Redis is not loaded yet causing 500 error: {e}")
            } else {
                error!("Redis error: {e}");
            }
            StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let mut word1 = payload.word1;
    let mut word2 = payload.word2;
    word1.vector = Some(vec1);
    word2.vector = Some(vec2);

    let sim = calculate_similarity(
        word1.vector.as_ref().unwrap(),
        word2.vector.as_ref().unwrap(),
        payload.measure
    );


    let res = SimilarityResponse { 
        similarity: sim, 
        words: [word1, word2] 
        };

    Ok(Json(res))
}



#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    info!("Logger initialised");

    let mut conn = get_redis_conn_with_retries().await?;
    let db_ready: bool = redis_exists_with_retries(&mut conn, "GLOVE:OK").await?;
    if db_ready {
        info!("Embeddings already in Redis, skipping streaming.");
    } else {
        info!("Streaming embeddings into Redis...");
        init_db(&mut conn).await?;
        info!("Finished streaming embeddings.");
    }


    println!("Server starting at :3000");
    
    let app: axum::Router = Router::new().route("/similarity", post(similarity_handler));

    let listener = TcpListener::bind("localhost:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();  

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

// calculates the dot product between 2 vectors `vec1` and `vec2`
fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length");
    vec1.iter().zip(vec2).map(|(a, b)| a * b).sum()
}


/// Calculates the cosine simuilarity between 2 vectors `vec1` and `vec2`.
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot = dot_product(vec1, vec2);
    let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm1 * norm2)
}


fn calculate_similarity(vec1: &[f32], vec2: &[f32], measure: Measure) -> f32 {
    match measure {
        Measure::Cosine => cosine_similarity(vec1, vec2),
        Measure::Dot => dot_product(vec1, vec2)
    }
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

/// Retrieves a connection to Redis,  and has a retry loop that retries `REDIS_RETRY_COUNT` amount of times with a `REDIS_RETRY_DELAY_MS` wait. This does this whenever Redis is still busy loading (if Redis crashes or is initialising still)
async fn get_redis_conn_with_retries() -> Result<MultiplexedConnection, RedisError> {
    info!("Attempting to retrieve Redis connection in {REDIS_MAX_RETRY_COUNT} retries");
    for attempt in 0..=REDIS_MAX_RETRY_COUNT {
        match REDIS_CLIENT.get_multiplexed_async_connection().await {
            Ok(c) => return Ok(c),
            Err(e) if e.kind() == redis::ErrorKind::BusyLoadingError => {
                warn!("Redis is loading (attempt {}): {e}", attempt + 1);
                tokio::time::sleep(Duration::from_millis(REDIS_RETRY_DELAY_MS)).await;
            }
            Err(e) => {
                error!("Redis error occured: {e}");
                return Err(e)
            },
        }
    }
    Err(RedisError::from((redis::ErrorKind::BusyLoadingError, "Redis still loading")))
}

/// Checks if key exists in Redis db connection, and has a retry loop that retries `REDIS_MAX_RETRY_COUNT` amount of times with a `REDIS_RETRY_DELAY_MS` wait. This does this whenever Redis is still busy loading (if Redis crashes or is initialising still)
async fn redis_exists_with_retries(
    conn: &mut MultiplexedConnection,
    key: &str,
) -> Result<bool, RedisError> {
    for attempt in 0..=REDIS_MAX_RETRY_COUNT {
        match conn.exists(key).await {
            Ok(val) => return Ok(val),
            Err(e) if e.kind() == redis::ErrorKind::BusyLoadingError => {
                warn!("Redis is still loading (attempt {}): {e}", attempt + 1);
                tokio::time::sleep(Duration::from_millis(REDIS_RETRY_DELAY_MS)).await;
            }
            Err(e) => {
                error!("Redis error occured: {e}");
                return Err(e)
            },        }
    }

    Err(RedisError::from((
        redis::ErrorKind::BusyLoadingError,
        "Redis still loading after retries",
    )))
}