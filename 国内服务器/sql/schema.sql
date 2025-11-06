CREATE TABLE IF NOT EXISTS query_results (
    task_id VARCHAR(64) PRIMARY KEY,
    answer TEXT,
    status ENUM('pending', 'done', 'error') DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
