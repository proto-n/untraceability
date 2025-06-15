sqlite3 zcash.sqlite <<"EOF"
.mode tabs
.output sqlite_outputs/zcash_shielded_input_distribution.tsv
select
    transactions_dedup.shielded_value_delta
from (
    select distinct time, hash, shielded_value_delta from transactions
) as transactions_dedup
where
    date(transactions_dedup.time) < date("2021-02-01 00:00:00") and
    transactions_dedup.shielded_value_delta > 0;
EOF
gzip -f sqlite_outputs/zcash_shielded_input_distribution.tsv
