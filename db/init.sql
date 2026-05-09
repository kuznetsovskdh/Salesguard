CREATE TABLE IF NOT EXISTS uploads (
    id          SERIAL PRIMARY KEY,
    filename    VARCHAR(255) NOT NULL,
    format      VARCHAR(20),
    encoding    VARCHAR(20),
    rows_total  INTEGER,
    rows_clean  INTEGER,
    status      VARCHAR(20) DEFAULT 'pending',
    error_msg   TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sales (
    id           SERIAL PRIMARY KEY,
    upload_id    INTEGER REFERENCES uploads(id),
    sale_date    DATE,
    product      VARCHAR(255),
    quantity     NUMERIC(12,2),
    unit_price   NUMERIC(12,2),
    revenue      NUMERIC(14,2),
    customer_id  VARCHAR(100),
    region       VARCHAR(100),
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS column_mapping (
    id             SERIAL PRIMARY KEY,
    upload_id      INTEGER REFERENCES uploads(id),
    source_col     VARCHAR(255),
    mapped_role    VARCHAR(50),
    confidence     NUMERIC(5,2),
    user_confirmed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS value_verdicts (
    id          SERIAL PRIMARY KEY,
    upload_id   INTEGER REFERENCES uploads(id),
    column_name VARCHAR(255),
    row_index   INTEGER,
    is_valid    BOOLEAN,
    reason      VARCHAR(255)
);

CREATE INDEX idx_sales_upload ON sales(upload_id);
CREATE INDEX idx_sales_date ON sales(sale_date);
CREATE INDEX idx_verdicts_upload ON value_verdicts(upload_id);
