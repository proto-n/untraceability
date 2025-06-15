sqlite3 bitcoin.sqlite <<"EOF"
CREATE TEMPORARY TABLE
	interest_filter
AS 
    SELECT DISTINCT recipient
    FROM
        inputs
    WHERE
        date(spending_time) >= date("2021-02-01 00:00:00")
        AND
        date(spending_time) < date("2021-02-08 00:00:00")
UNION 
    SELECT DISTINCT recipient
    FROM
        outputs
    WHERE
        date(time) >= date("2021-02-01 00:00:00")
        AND
        date(time) < date("2021-02-08 00:00:00");

CREATE TEMPORARY TABLE
	inputs_dedup
AS 
    SELECT DISTINCT
        inputs.transaction_hash, inputs.ix, inputs.value, inputs.recipient
    FROM
        inputs 
            INNER JOIN interest_filter
            ON inputs.recipient = interest_filter.recipient
    WHERE
        date(spending_time) < date("2021-02-01 00:00:00");

.mode tabs
.output sqlite_outputs/bitcoin_inputs_sum.tsv
SELECT
    recipient,
    sum(value)
FROM
    inputs_dedup
GROUP BY recipient;
EOF
gzip -f sqlite_outputs/bitcoin_inputs_sum.tsv

sqlite3 bitcoin.sqlite <<"EOF"
CREATE TEMPORARY TABLE
	interest_filter
AS 
    SELECT DISTINCT recipient
    FROM
        inputs
    WHERE
        date(spending_time) >= date("2021-02-01 00:00:00")
        AND
        date(spending_time) < date("2021-02-08 00:00:00")
UNION 
    SELECT DISTINCT recipient
    FROM
        outputs
    WHERE
        date(time) >= date("2021-02-01 00:00:00")
        AND
        date(time) < date("2021-02-08 00:00:00");

CREATE TEMPORARY TABLE
	outputs_dedup
AS 
    SELECT DISTINCT
        outputs.transaction_hash, outputs.ix, outputs.value, outputs.recipient
    FROM
        outputs
            INNER JOIN interest_filter
            ON outputs.recipient = interest_filter.recipient
    WHERE
        date(time) < date("2021-02-01 00:00:00");

.mode tabs
.output sqlite_outputs/bitcoin_outputs_sum.tsv
SELECT
    recipient,
    sum(value)
FROM
    outputs_dedup
GROUP BY recipient;
EOF
gzip -f sqlite_outputs/bitcoin_outputs_sum.tsv
