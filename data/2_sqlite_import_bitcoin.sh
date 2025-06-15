zcat_newlines () {
	for i in "$@"
	do {
		zcat $i;
		echo;
	} done
}

sqlite3 bitcoin.sqlite <<"EOF"
create table blocks (
	id integer,
	hash text,
	time text,
	median_time text,
	size int,
	stripped_size int,
	weight int,
	version text,
	version_hex text,
	version_bits int,
	merkle_root text,
	nonce text,
	bits int,
	difficulty int,
	chainwork text,
	coinbase_data_hex text,
	transaction_count int,
	witness_count int,
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
	fee_per_kwu real,
	fee_per_kwu_usd real,
	cdd_total real,
	generation int,
	generation_usd real,
	reward int,
	reward_usd real,
	guessed_miner text
);
EOF
zcat_newlines bitcoin/blocks/*.gz | grep -v median_time | sqlite3 bitcoin.sqlite ".mode tabs"  "BEGIN TRANSACTION" ".import /dev/stdin blocks" "COMMIT"

sqlite3 bitcoin.sqlite <<"EOF"
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
	spending_witness text,
	lifespan int,
	cdd real
);
EOF
zcat_newlines bitcoin/inputs/*.gz | grep -v transaction_hash | sqlite3 bitcoin.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin inputs" "COMMIT"

sqlite3 bitcoin.sqlite <<"EOF"
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
)
EOF
zcat_newlines bitcoin/outputs/*.gz | grep -v transaction_hash | sqlite3 bitcoin.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin outputs" "COMMIT"

sqlite3 bitcoin.sqlite <<"EOF"
create table transactions (
	block_id int,
	hash text,
	time text,
	size int,
	weight int,
	version text,
	lock_time int,
	is_coinbase int,
	has_witness int,
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
	fee_per_kwu real,
	fee_per_kwu_usd real,
	cdd_total real
)
EOF
zcat_newlines bitcoin/transactions/*.gz | grep -v block_id | sqlite3 bitcoin.sqlite ".mode tabs"  "BEGIN TRANSACTION" ".import /dev/stdin transactions" "COMMIT"