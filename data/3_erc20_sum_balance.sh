sqlite3 erc20.sqlite <<"EOF"
.load ./decimal
.mode tabs
.output sqlite_outputs/erc20_inputs_sum.tsv
select
    sender,
    token_address,
    decimal_sum(value)
from transactions
where
    date(time) < date("2021-02-01 00:00:00")
group by sender, token_address;
EOF
gzip sqlite_outputs/erc20_inputs_sum.tsv

sqlite3 erc20.sqlite <<"EOF"
.load ./decimal
.mode tabs
.output sqlite_outputs/erc20_outputs_sum.tsv
select
    recipient,
    token_address,
    decimal_sum(value)
from transactions
where
    date(time) < date("2021-02-01 00:00:00")
group by recipient, token_address;
EOF
gzip sqlite_outputs/erc20_outputs_sum.tsv 