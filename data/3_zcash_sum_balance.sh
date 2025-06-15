sqlite3 zcash.sqlite <<"EOF"
.mode tabs
.output sqlite_outputs/zcash_inputs_sum.tsv
select
    inputs_dedup.recipient,
    sum(inputs_dedup.value)
from (
    select distinct time, spending_time, transaction_hash, ix, recipient, value from inputs
) as inputs_dedup
where
    date(inputs_dedup.spending_time) < date("2021-02-01 00:00:00")
group by recipient;
EOF
gzip -f sqlite_outputs/zcash_inputs_sum.tsv

sqlite3 zcash.sqlite <<"EOF"
.mode tabs
.output sqlite_outputs/zcash_outputs_sum.tsv
select
    outputs_dedup.recipient,
    sum(outputs_dedup.value)
from (
    select distinct time, transaction_hash, ix, recipient, value from outputs
) as outputs_dedup
where
    date(outputs_dedup.time) < date("2021-02-01 00:00:00")
group by recipient;
EOF
gzip -f sqlite_outputs/zcash_outputs_sum.tsv

sqlite3 zcash.sqlite <<"EOF"
.mode tabs
.output sqlite_outputs/zcash_shielded_sum.tsv
select
    sum(transactions_dedup.shielded_value_delta)
from (
    select distinct time, hash, shielded_value_delta from transactions
) as transactions_dedup
where
    date(transactions_dedup.time) < date("2021-02-01 00:00:00");
EOF
gzip -f sqlite_outputs/zcash_shielded_sum.tsv
