# You should pass the Blockchair API key as the first argument to this script.
KEY=$1
BASEURL="https://gz.blockchair.com/"
DIRS=$(cat dirs)
for DIR in $DIRS
do
    mkdir -p $DIR
    pushd $DIR
    GZ_FILES=$(curl --insecure -sNL ${BASEURL}${DIR}/  | grep ".gz\"" | sed 's/^.*"\([^"]*\.gz\)".*$/\1/g')
    for F in $GZ_FILES
    do
        wget --no-check-certificate -c ${BASEURL}${DIR}/${F}?key=$KEY -O ${F}
    done
    popd
done
