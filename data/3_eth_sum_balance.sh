# You are going to need the decimal extension for sqlite: https://github.com/nalgeon/sqlean/issues/27#issuecomment-1007348326

sqlite3 ethereum.sqlite <<"EOF"
.load ./decimal
.mode tabs
.output sqlite_outputs/eth_fee_sum.tsv
select
    sender,
    decimal_sum(fee)
from transactions
where
    date(time) < date("2021-02-01 00:00:00")
group by sender;
EOF
gzip -f sqlite_outputs/eth_fee_sum.tsv

sqlite3 ethereum.sqlite <<"EOF"
.load ./decimal
.mode tabs
.output sqlite_outputs/eth_call_recipient.tsv
select
    recipient,
    decimal_sum(value)
from calls
where
    date(time) < date("2021-02-01 00:00:00") and
    failed=0
group by recipient;
EOF
gzip -f sqlite_outputs/eth_call_recipient.tsv

sqlite3 ethereum.sqlite <<"EOF"
.load ./decimal
.mode tabs
.output sqlite_outputs/eth_call_sender.tsv
select
    sender,
    decimal_sum(value)
from calls
where
    date(time) < date("2021-02-01 00:00:00") and
    failed=0
group by sender;
EOF
gzip -f sqlite_outputs/eth_call_sender.tsv
