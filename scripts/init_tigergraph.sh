#!/usr/bin/env bash
set -euo pipefail

USER=${USER:-tigergraph}
PASS=${PASSWORD:-tigergraph}
GRAPH=${TG_GRAPH:-DocGraph}
TOKEN_TTL=${TG_TOKEN_TTL:-2592000}
LICENSE_PATH=/home/tigergraph/enterprise-license.txt

su - tigergraph -c "gadmin start all" || true
sleep 5

/opt/scripts/wait_for_tg.sh || true

if [[ -f "$LICENSE_PATH" ]]; then
  su - tigergraph -c "gadmin license set -f $LICENSE_PATH || true"
  su - tigergraph -c "gadmin restart all || true"
  sleep 5
fi

su - tigergraph -c "echo -e '${PASS}\n${PASS}' | passwd tigergraph || true"
su - tigergraph -c "gsql 'CREATE GRAPH ${GRAPH}()'" || true

TMP_SCHEMA=$(mktemp)
TMP_QUERIES=$(mktemp)
sed "s|\$TG_GRAPH|${GRAPH}|g" /opt/gsql/schema.gsql > "$TMP_SCHEMA"
sed "s|\$TG_GRAPH|${GRAPH}|g" /opt/gsql/queries.gsql > "$TMP_QUERIES"

su - tigergraph -c "gsql $TMP_SCHEMA" || true
su - tigergraph -c "gsql $TMP_QUERIES" || true

rm -f "$TMP_SCHEMA" "$TMP_QUERIES"
