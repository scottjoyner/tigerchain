#!/usr/bin/env bash
set -euo pipefail

PASS="${PASSWORD:-tigergraph}"
GRAPH="${TG_GRAPH:-DocGraph}"

# Ensure OS password
echo "tigergraph:${PASS}" | chpasswd || true

# Ensure TG env for the user
ln -sf /home/tigergraph/tigergraph/configs/tg.cfg /home/tigergraph/.tg.cfg
chown -h tigergraph:tigergraph /home/tigergraph/.tg.cfg
grep -q "/home/tigergraph/tigergraph/app/cmd" /home/tigergraph/.bashrc || \
  (echo 'export PATH=$PATH:/home/tigergraph/tigergraph/app/cmd' >> /home/tigergraph/.bashrc && chown tigergraph:tigergraph /home/tigergraph/.bashrc)

# Start services (idempotent)
su - tigergraph -c "gadmin start all" || true
sleep 5

# Create graph and install schema/loaders/queries
su - tigergraph -c "gsql \"create graph ${GRAPH}()\"" || true
su - tigergraph -c "gsql -g ${GRAPH} /opt/gsql/schema.gsql"
su - tigergraph -c "gsql -g ${GRAPH} /opt/gsql/loading_jobs.gsql"
su - tigergraph -c "gsql -g ${GRAPH} /opt/gsql/queries.gsql"

echo "[init_tigergraph] Graph '${GRAPH}' initialized."
