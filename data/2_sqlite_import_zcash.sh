zcat_newlines () {
	for i in "$@"
	do {
		zcat $i;
		echo;
	} done
}

sqlite3 zcash.sqlite <<"EOF"
create table blocks (
	id int,
	hash text,
	time text,
	size int,
	version int,
	version_hex int,
	version_bits int,
	merkle_root text,
	final_sapling_root int,
	nonce int,
	solution text,
	anchor text,
	bits int,
	difficulty real,
	chainwork text,
	coinbase_data_hex text,
	transaction_count int,
	input_count int,
	output_count int,
	input_total int,
	input_total_usd real,
	output_total int,
	output_total_usd real,
	fee_total int,
	fee_total_usd real,
	fee_per_kb real,
	fee_per_kb_usd real,
	cdd_total real,
	generation int,
	generation_usd real,
	reward int,
	reward_usd real,
	guessed_miner text,
	shielded_value_delta_total int
);
EOF
zcat_newlines zcash/blocks/*.gz | grep -v final_sapling_root | sqlite3 zcash.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin blocks" "COMMIT"

sqlite3 zcash.sqlite <<"EOF"
create table inputs (
	block_id int,
	transaction_hash text,
	ix int,
	time text,
	value int,
	value_usd real,
	recipient text,
	type text,
	script_hex text,
	is_from_coinbase int,
	is_spendable int,
	spending_block_id int,
	spending_transaction_hash text,
	spending_index int,
	spending_time text,
	spending_value_usd real,
	spending_sequence int,
	spending_signature_hex text,
	lifespan int,
	cdd real
);
EOF
zcat_newlines zcash/inputs/*.gz | grep -v transaction_hash | sqlite3 zcash.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin inputs" "COMMIT"

sqlite3 zcash.sqlite <<"EOF"
create table outputs (
	block_id int,
	transaction_hash text,
	ix int,
	time text,
	value int,
	value_usd real,
	recipient text,
	type text,
	script_hex text,
	is_from_coinbase int,
	is_spendable int
);
EOF
zcat_newlines zcash/outputs/*.gz | grep -v transaction_hash | sqlite3 zcash.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin outputs" "COMMIT"


sqlite3 zcash.sqlite <<"EOF"
create table transactions (
	block_id int,
	hash text,
	time text,
	size int,
	is_overwintered int,
	version int,
	version_group_id text,
	expiry_height text,
	lock_time int,
	is_coinbase int,
	input_count int,
	output_count int,
	input_total int,
	input_total_usd real,
	output_total int,
	output_total_usd real,
	fee int,
	fee_usd real,
	fee_per_kb real,
	fee_per_kb_usd real,
	cdd_total real,
	shielded_value_delta int,
	join_split_raw text,
	shielded_input_raw text,
	shielded_output_raw text,
	binding_signature text
);
EOF
zcat_newlines zcash/transactions/*.gz | grep -v block_id | sqlite3 zcash.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin transactions" "COMMIT"