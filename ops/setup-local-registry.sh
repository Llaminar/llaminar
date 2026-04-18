#!/bin/bash
# Set up a disk-backed Docker registry on the local host so layer cache
# survives reboots (which wipe the tmpfs-backed /var/lib/docker).
#
# - Storage: /var/lib/llaminar-registry/data  (on the rootfs nvme, NOT tmpfs)
# - Port:    5000 on all interfaces (firewall already restricts to LAN)
# - Lifecycle: systemd unit `llaminar-registry.service`, restarts on reboot
# - Trust: dockerd configured to allow http for localhost:5000 + LAN host name
set -euo pipefail

# `xeon` is the physical runner box on the LAN (192.168.50.98). DO NOT use
# `home.sanftenberg.net` here — that resolves to the router downstairs and
# would force every registry round-trip out to wifi and back. CI itself uses
# `localhost:5000` so it's pure loopback; this LAN entry is only here so
# other LAN hosts (devboxes etc.) can also pull from the registry.
REG_HOST="xeon"
REG_PORT=5000
DATA_DIR="/var/lib/llaminar-registry/data"
UNIT="/etc/systemd/system/llaminar-registry.service"
DOCKER_CFG="/etc/docker/daemon.json"

echo "==> Preparing storage"
sudo install -d -m 0755 -o root -g root "$DATA_DIR"

echo "==> Pulling registry:2"
sudo docker pull registry:2

echo "==> Removing any prior container"
sudo docker rm -f llaminar-registry 2>/dev/null || true

echo "==> Writing systemd unit"
sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=Llaminar local Docker registry (disk-backed BuildKit cache)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker rm -f llaminar-registry
ExecStart=/usr/bin/docker run --rm --name llaminar-registry \\
    -p ${REG_PORT}:5000 \\
    -v ${DATA_DIR}:/var/lib/registry \\
    -e REGISTRY_STORAGE_DELETE_ENABLED=true \\
    -e REGISTRY_HTTP_HEADERS_X-Content-Type-Options="[nosniff]" \\
    registry:2
ExecStop=/usr/bin/docker stop llaminar-registry

[Install]
WantedBy=multi-user.target
EOF

echo "==> Configuring dockerd to trust the local registry over HTTP"
# Merge insecure-registries into daemon.json without clobbering other keys.
sudo install -d -m 0755 /etc/docker
if [ ! -f "$DOCKER_CFG" ]; then
    echo '{}' | sudo tee "$DOCKER_CFG" >/dev/null
fi
TMP=$(mktemp)
sudo cat "$DOCKER_CFG" | python3 -c "
import json, sys
cfg = json.load(sys.stdin) if sys.stdin.read else {}
" 2>/dev/null || true
# Simpler: use jq.
if ! command -v jq >/dev/null; then
    sudo apt-get install -y jq
fi
# Replace any prior insecure-registries list (we want to drop stale entries
# like `home.sanftenberg.net:5000` from earlier runs that pointed at the
# router instead of the host).
sudo jq --arg h1 "localhost:${REG_PORT}" --arg h2 "127.0.0.1:${REG_PORT}" --arg h3 "${REG_HOST}:${REG_PORT}" \
    '. + {"insecure-registries": [$h1, $h2, $h3]}' \
    "$DOCKER_CFG" > "$TMP"
sudo mv "$TMP" "$DOCKER_CFG"
sudo chmod 0644 "$DOCKER_CFG"
echo "--- daemon.json ---"
sudo cat "$DOCKER_CFG"

echo "==> Reloading dockerd to pick up insecure-registries"
sudo systemctl reload docker || sudo systemctl restart docker

echo "==> Enabling + starting llaminar-registry"
sudo systemctl daemon-reload
sudo systemctl enable --now llaminar-registry

sleep 3
echo "==> Verifying"
sudo systemctl --no-pager status llaminar-registry | head -15
curl -fsS "http://localhost:${REG_PORT}/v2/" && echo " <- /v2/ OK"
echo
echo "==> Smoke test: tag + push + pull busybox"
sudo docker pull busybox:latest >/dev/null
sudo docker tag busybox:latest "localhost:${REG_PORT}/busybox-test:1"
sudo docker push "localhost:${REG_PORT}/busybox-test:1"
sudo docker rmi "localhost:${REG_PORT}/busybox-test:1" >/dev/null
sudo docker pull "localhost:${REG_PORT}/busybox-test:1"
sudo docker rmi "localhost:${REG_PORT}/busybox-test:1" >/dev/null
echo "==> Smoke test passed"

echo
echo "==> Storage usage"
sudo du -sh "$DATA_DIR"
df -h "$DATA_DIR"
