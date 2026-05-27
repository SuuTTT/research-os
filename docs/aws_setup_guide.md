# AWS Setup Guide

Date: 2026-05-25

This guide sets up a brand-new AWS control plane for Research OS. The EC2
instance is the master/coordinator. It does not need a GPU.

The master runs:
- project manifests;
- idea queue;
- benchmark queue;
- run queue;
- publication queue;
- worker registry;
- docs/templates;
- S3 backups;
- future dashboard and agents.

Workers are separate machines: VastAI GPUs, EC2 GPUs, local machines, CPU
analysis boxes, or cluster jobs.

## 1. Launch EC2

On the EC2 launch page:

- Name: `research-os-master`
- AMI: Ubuntu Server 24.04 LTS or 22.04 LTS
- Architecture:
  - choose `x86_64` if you may build Docker images on this box;
  - choose ARM only if it is strictly a control-plane machine.
- Instance type:
  - minimum: `t3.small`;
  - better default: `t3.medium`;
  - if many dashboards/agents: `t3.large`.
- Key pair:
  - create or select one;
  - download the `.pem`;
  - do not lose it.
- Storage:
  - root EBS: `100 GB` minimum;
  - better: `200-500 GB gp3`.
- Security group inbound:
  - SSH `22/tcp` from your IP only;
  - dashboard later: `8080/tcp` or `5056/tcp` from your IP only, or keep closed
    and use an SSH tunnel.

Do not open SSH or dashboard ports to `0.0.0.0/0`.

## 2. SSH In

From your local machine:

```bash
chmod 600 ~/Downloads/your-key.pem
ssh -i ~/Downloads/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

Become root:

```bash
sudo -i
```

## 3. Install Base Packages

```bash
apt-get update
apt-get install -y \
  git curl wget unzip jq tmux htop ripgrep rsync \
  python3 python3-venv python3-pip \
  build-essential ca-certificates gnupg
```

Optional Docker:

```bash
apt-get install -y docker.io
systemctl enable --now docker
```

## 4. Install AWS CLI

For `x86_64` Ubuntu:

```bash
cd /tmp
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip
./aws/install
aws --version
```

Configure credentials if you want S3 backups:

```bash
aws configure
```

Use an IAM user or role with limited S3 permissions. Do not use AWS root
credentials.

## 5. Install Research OS

If copying from an existing machine:

```bash
mkdir -p /root/research-os
```

From the source machine:

```bash
rsync -az /root/research-os/ root@<EC2_PUBLIC_IP>:/root/research-os/
```

If using GitHub:

```bash
cd /root
git clone <YOUR_RESEARCH_OS_REPO_URL> research-os
```

## 6. Create Python Environment

```bash
cd /root/research-os
python3 -m venv /root/research-os/.venv
source /root/research-os/.venv/bin/activate
pip install --upgrade pip
pip install pyyaml flask rich
```

The current CLI uses only the Python standard library, but these packages are
useful for the next dashboard and agent layers.

Test:

```bash
python3 scripts/ros.py status
python3 scripts/ros.py list-ideas
```

Install VastAI CLI (for GPU worker hunting/visibility):

```bash
pip3 install vastai --break-system-packages
export PATH="$HOME/.local/bin:$PATH"
vastai --version
```

## 6.1 Add Runtime Credentials

Store runtime provider keys locally on the master:

```bash
cat > /home/ubuntu/.env.local <<'EOF'
WANDB_API_KEY=...
HF_TOKEN=...
GITHUB_TOKEN=...
VASTAI_API_KEY=...
EOF
chmod 600 /home/ubuntu/.env.local
```

Auto-load on login:

```bash
grep -q 'env.local' ~/.bashrc || cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
set -a; source /home/ubuntu/.env.local; set +a
EOF
```

Credential smoke tests:

```bash
source /home/ubuntu/.env.local
vastai show instances
curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" https://api.github.com/user
curl -sS -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2
curl -sS -u "api:$WANDB_API_KEY" -H "Content-Type: application/json" \
  -X POST https://api.wandb.ai/graphql \
  --data '{"query":"query { viewer { username } }"}'
```

Never commit `.env.local` or API keys to Git.

## 7. SSH Key For Workers

The master needs an SSH key for future workers.

```bash
ssh-keygen -t ed25519 -f /root/.ssh/research_os_ed25519 -N ""
cat /root/.ssh/research_os_ed25519.pub
```

For every worker, add the public key to:

```text
/root/.ssh/authorized_keys
```

Keep the private key only on the master.

## 8. S3 Backup Bucket

In AWS Console:

1. Go to S3.
2. Create bucket, for example `research-os-control-<yourname>`.
3. Keep public access blocked.
4. Enable versioning if budget allows.

Test upload:

```bash
echo ok > /tmp/research-os-test.txt
aws s3 cp /tmp/research-os-test.txt s3://YOUR_BUCKET/test.txt
```

## 9. Add Control-Plane Backup Script

Create:

```bash
cat > /root/research-os/scripts/backup_control_plane.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/research-os}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=/root/research-os/backups
mkdir -p "$OUT"

cd "$ROOT"
tar -czf "$OUT/research_os_control_${STAMP}.tgz" \
  queues docs schemas templates scripts workers \
  research/*/project.yaml research/*/deep_research research/*/paper research/*/blog \
  2>/dev/null || true

ln -sf "research_os_control_${STAMP}.tgz" "$OUT/research_os_control_latest.tgz"

if [[ -n "${S3_URI:-}" ]]; then
  aws s3 cp "$OUT/research_os_control_${STAMP}.tgz" "$S3_URI/"
  aws s3 cp "$OUT/research_os_control_${STAMP}.tgz" "$S3_URI/research_os_control_latest.tgz"
fi
EOF

chmod +x /root/research-os/scripts/backup_control_plane.sh
```

Run:

```bash
S3_URI=s3://YOUR_BUCKET/control bash /root/research-os/scripts/backup_control_plane.sh
```

Add cron:

```bash
crontab -e
```

Add:

```cron
*/15 * * * * S3_URI=s3://YOUR_BUCKET/control bash /root/research-os/scripts/backup_control_plane.sh >> /root/research-os/backups/backup.log 2>&1
```

## 10. Optional Elastic IP

If you want a stable IP:

1. EC2 Console -> Elastic IPs.
2. Allocate Elastic IP.
3. Associate it to `research-os-master`.

Release it when you no longer need it. Elastic IPs can cost money even when
idle.

## 11. Create First Project

Example:

```bash
cd /root/research-os

python3 scripts/ros.py init-project \
  --project-id structural_entropy_timeseries \
  --title "Structural entropy for neural time-series forecasting" \
  --idea "Integrate structural entropy into N-BEATS, N-BEATSx, and TimesNet to improve long-horizon forecasting." \
  --target "Beat strong reported baselines on ETT, Weather, Traffic, and Electricity under fair MSE/MAE evaluation." \
  --metric "MSE/MAE" \
  --priority 5
```

Then:

```bash
python3 scripts/ros.py status
python3 scripts/ros.py list-ideas
```

## 12. Deep Research Step

Open:

```text
/root/research-os/research/structural_entropy_timeseries/deep_research/request.md
```

Paste it into GPT Deep Research or another literature tool.

Save outputs:

```text
/root/research-os/research/structural_entropy_timeseries/deep_research/report.md
/root/research-os/research/structural_entropy_timeseries/deep_research/sota_table.csv
/root/research-os/research/structural_entropy_timeseries/deep_research/refs.bib
```

## 13. GitHub Setup

On EC2:

```bash
cd /root/research-os
git init
git add README.md docs schemas scripts templates workers queues \
  research/structural_entropy_timeseries/project.yaml \
  research/structural_entropy_timeseries/deep_research/request.md
git commit -m "Initial research OS scaffold"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Do not commit datasets, checkpoints, W&B logs, large benchmark repos, or
artifacts.

## 14. Next Build Steps

After the EC2 control plane is stable:

1. Add a simple Flask dashboard reading `/root/research-os/queues/*.json`.
2. Add worker registry support from `workers/workers.yaml`.
3. Add a local run daemon for `run_queue.json`.
4. Add SSH worker launcher.
5. Add benchmark-agent scripts:
   - clone repo;
   - create env;
   - smoke test;
   - parse metric.
6. Add publication generator:
   - blog from queue evidence;
   - LaTeX manuscript update.

## Immediate Health Check

```bash
cd /root/research-os
python3 scripts/ros.py status
bash scripts/backup_control_plane.sh
aws s3 ls s3://YOUR_BUCKET/control/
```

Once these pass, the EC2 master is ready as the persistent Research OS control
plane.

