zcat_newlines () {
	for i in "$@"
	do {
		zcat $i;
		echo;
	} done
}

sqlite3 erc20.sqlite <<"EOF"
create table tokens (
	id int,
	address text,
	time text,
	name text,
	symbol text,
	decimals int,
	creating_block_id int,
	creating_transaction_hash text
);
EOF
zcat_newlines ethereum/erc-20/tokens/*.gz | grep -v address | sed "s/\"//g" | sqlite3 erc20.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin tokens" "COMMIT"

sqlite3 erc20.sqlite <<"EOF"
create table transactions (
	block_id int,
	transaction_hash text,
	time text,
	token_address text,
	token_name text,
	token_symbol text,
	token_decimals int,
	sender text,
	recipient text,
	value text
);
EOF
zcat_newlines ethereum/erc-20/transactions/*.gz | grep -v transaction_hash | sed "s/\"//g" | sqlite3 erc20.sqlite ".mode tabs" "BEGIN TRANSACTION" ".import /dev/stdin transactions" "COMMIT"
