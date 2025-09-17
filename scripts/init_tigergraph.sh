#!/usr/bin/env bash
set -euo pipefail

USER=${USER:-tigergraph}
PASS=${PASSWORD:-tigergraph}
GRAPH=${TG_GRAPH:-DocGraph}

su - tigergraph -c "gadmin start all" || true
sleep 5
/opt/scripts/wait_for_tg.sh || true

su - tigergraph -c "echo -e '${PASS}\n${PASS}' | passwd tigergraph || true"
su - tigergraph -c "gsql 'create graph $GRAPH()'" || true

su - tigergraph -c "gsql -g $GRAPH /opt/gsql/schema.gsql"
su - tigergraph -c "gsql -g $GRAPH /opt/gsql/loading_jobs.gsql"
su - tigergraph -c "gsql -g $GRAPH /opt/gsql/queries.gsql"
