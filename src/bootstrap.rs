use crate::config::Config;
use crate::database::DatabaseClient;
use crate::embeddings::{get_embedding_provider, EmbeddingProvider};
use anyhow::Result;

pub async fn init_database_and_provider(
    database_type: Option<String>,
    database_url: Option<String>,
    embedding_dim_override: Option<usize>,
) -> Result<(Config, DatabaseClient, Box<dyn EmbeddingProvider + Send + Sync>)> {
    let mut config = Config::from_env();
    
    if let Some(dim) = embedding_dim_override {
        config.embedding_dim = Some(dim);
    }
    
    if let Some(db_type) = database_type {
        let db_type_norm = db_type.to_lowercase();
        let db_url = database_url
            .or_else(|| {
                if (db_type_norm.as_str() == "sqlite" || db_type_norm.as_str() == "sqlite3") 
                    && config.database_url.starts_with("sqlite://") {
                    Some(config.database_url.clone())
                } else if (db_type_norm.as_str() == "postgresql" || db_type_norm.as_str() == "postgres" || db_type_norm.as_str() == "pg")
                    && (config.database_url.starts_with("postgresql://") || config.database_url.starts_with("postgres://")) {
                    Some(config.database_url.clone())
                } else {
                    None
                }
            })
            .or_else(|| std::env::var("MEMENTO_DATABASE_URL").ok())
            .or_else(|| {
                match db_type_norm.as_str() {
                    "postgresql" | "postgres" | "pg" => {
                        let url = std::env::var("DATABASE_URL").ok();
                        if url.is_none() {
                            eprintln!("⚠️  PostgreSQL requires a database URL. Set DATABASE_URL or MEMENTO_DATABASE_URL environment variable, or use --database-url");
                        }
                        url
                    }
                    _ => {
                        eprintln!("ℹ️  No database URL provided, defaulting to ./memento.db");
                        Some("./memento.db".to_string())
                    }
                }
            });
        
        let final_db_url = match db_type_norm.as_str() {
            "postgresql" | "postgres" | "pg" => {
                let url = db_url.ok_or_else(|| {
                    anyhow::anyhow!("PostgreSQL requires database URL")
                })?;
                if !url.starts_with("postgresql://") && !url.starts_with("postgres://") {
                    format!("postgresql://{}", url)
                } else {
                    url
                }
            }
            "sqlite" | "sqlite3" => {
                let path = db_url.unwrap_or_else(|| "./memento.db".to_string());
                DatabaseClient::normalize_sqlite_url(&path)
            }
            other => {
                eprintln!("⚠️  Unknown database_type '{}', defaulting to sqlite", other);
                let path = db_url.unwrap_or_else(|| "./memento.db".to_string());
                DatabaseClient::normalize_sqlite_url(&path)
            }
        };
        config.database_url = final_db_url;
    } else if let Some(db_url) = database_url {
        config.database_url = db_url;
    }
    
    let model = Some(config.embedding_model.clone());
    let embedding_provider = get_embedding_provider(
        match config.embedding_provider {
            crate::config::EmbeddingProvider::OpenAi => "openai",
            crate::config::EmbeddingProvider::Local => "local",
        },
        config.openai_api_key.clone(),
        model,
        config.embedding_dim,
    )?;
    
    let inferred_dim = embedding_provider.dim();
    config.embedding_dim.get_or_insert(inferred_dim);
    let db_dim = config.embedding_dim;
    
    let db = DatabaseClient::new_with_dim(&config.database_url, db_dim).await?;
    
    Ok((config, db, embedding_provider))
}

