#!/bin/bash
# Run this once on the vast.ai instance to set up a local TURN server.
# Usage: bash setup_turn.sh YOUR_PUBLIC_IP [EXTERNAL_PORT]
# Example: bash setup_turn.sh 123.45.67.89 3478

set -e

PUBLIC_IP=${1:?Usage: $0 PUBLIC_IP [PORT]}
TURN_PORT=${2:-3478}
TURN_USER="noor"
TURN_PASS="noorpass$(shuf -i 1000-9999 -n1)"   # random suffix each install

echo "[TURN] Installing coturn..."
apt-get update -qq && apt-get install -y -qq coturn

echo "[TURN] Writing config..."
cat > /etc/turnserver.conf << EOF
listening-port=${TURN_PORT}
external-ip=${PUBLIC_IP}
realm=noor.local
server-name=noor-turn
lt-cred-mech
user=${TURN_USER}:${TURN_PASS}
no-tls
no-dtls
log-file=/var/log/coturn.log
simple-log
EOF

echo "TURNSERVER_ENABLED=1" > /etc/default/coturn

echo "[TURN] Starting coturn..."
pkill turnserver 2>/dev/null || true
turnserver -c /etc/turnserver.conf -o

sleep 1
if pgrep turnserver > /dev/null; then
    echo ""
    echo "=========================================="
    echo " TURN server running. Add these exports:"
    echo "=========================================="
    echo "export TURN_URLS=\"turn:${PUBLIC_IP}:${TURN_PORT}?transport=tcp,turn:${PUBLIC_IP}:${TURN_PORT}\""
    echo "export TURN_USERNAME=\"${TURN_USER}\""
    echo "export TURN_CREDENTIAL=\"${TURN_PASS}\""
    echo "=========================================="
else
    echo "[ERROR] coturn failed to start. Check /var/log/coturn.log"
fi
