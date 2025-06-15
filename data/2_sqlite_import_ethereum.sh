zcat_newlines () {
	for i in "$@"
	do {
		zcat $i;
		echo;
	} done
}

sqlite3 ethereum.sqlite <<"EOF"
create table blocks (
	id int,
	hash text,
	time text,
	size int,
	miner text,
	extra_data_hex text,
	difficulty int,
	gas_used text,
	gas_limit text,
	logs_bloom text,
	mix_hash text,
	nonce int,
	receipts_root text,
	sha3_uncles text,
	state_root text,
	total_difficulty int,
	transactions_root text,
	uncle_count int,
	transaction_count int,
	synthetic_transaction_count int,
	call_count int,
	synthetic_call_count int,
	value_total text,
	value_total_usd real,
	internal_value_total text,
	internal_value_total_usd real,
	generation text,
	generation_usd real,
	uncle_generation text,
	uncle_generation_usd real,
	fee_total text,
	fee_total_usd real,
	reward text,
	reward_usd real
);
EOF
zcat_newlines ethereum/blocks/*.gz | grep -v extra_data_hex | sqlite3 ethereum.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin blocks" "COMMIT"

sqlite3 ethereum.sqlite <<"EOF"
create table transactions (
	block_id int,
	ix int,
	hash text,
	time text,
	failed int,
	type text,
	sender text,
	recipient text,
	call_count int,
	value text,
	value_usd real,
	internal_value text,
	internal_value_usd real,
	fee text,
	fee_usd real,
	gas_used text,
	gas_limit text,
	gas_price text,
	input_hex text,
	nonce int,
	v text,
	r text,
	s text
);
EOF
zcat_newlines ethereum/transactions/*.gz | grep -v block_id | sqlite3 ethereum.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin transactions" "COMMIT"

sqlite3 ethereum.sqlite <<"EOF"
create table uncles (
	parent_block_id int,
	ix int,
	id int,
	hash text,
	time text,
	size int,
	miner text,
	extra_data_hex text,
	difficulty int,
	gas_used text,
	gas_limit text,
	logs_bloom text,
	mix_hash text,
	nonce int,
	receipts_root text,
	sha3_uncles text,
	state_root text,
	transactions_root text,
	generation text,
	generation_usd real
);
EOF
zcat_newlines ethereum/uncles/*.gz | grep -v block_id | sqlite3 ethereum.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin uncles" "COMMIT"

sqlite3 ethereum.sqlite <<"EOF"
create table calls (
	block_id int,
	transaction_hash text,
	ix int,
	depth int,
	time text,
	failed int,
	fail_reason text,
	type text,
	sender text,
	recipient text,
	child_call_count int,
	value text,
	value_usd real,
	transferred text,
	input_hex text,
	output_hex text
);
EOF
zcat_newlines ethereum/calls/*.gz | grep -v block_id | sqlite3 ethereum.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin calls" "COMMIT"