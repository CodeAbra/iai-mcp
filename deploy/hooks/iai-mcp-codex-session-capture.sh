#!/usr/bin/env bash
# IAI-MCP Codex Stop hook — ambient WRITE-side capture.
#
# Reads Codex's Stop-hook JSON payload from stdin, extracts the active
# transcript path, and defers capture through `iai-mcp capture-transcript
# --no-spawn`. Fail-safe by design: any error exits 0 so session teardown is
# never blocked.

set -u  # no -e: the hook must not block session teardown
input=$(cat 2>/dev/null || true)

extract() {
  local key=$1
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r ".${key} // empty" 2>/dev/null
  else
    printf '%s' "$input" | /usr/bin/python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('${key}', '') or '')
except Exception:
    print('')
" 2>/dev/null
  fi
}

session_id=$(extract "session_id")
transcript_path=$(extract "transcript_path")
if [[ -z "$transcript_path" ]]; then
  transcript_path=$(extract "transcriptPath")
fi
cwd=$(extract "cwd")
stop_hook_active=$(extract "stop_hook_active")

if [[ -z "$transcript_path" && -n "$session_id" ]]; then
  sessions_dir="$HOME/.codex/sessions"
  if [[ -d "$sessions_dir" ]]; then
    transcript_path=$(
      find "$sessions_dir" -type f \( -name "${session_id}.jsonl" -o -name "*${session_id}*.jsonl" \) \
        -print 2>/dev/null | head -n 1
    )
  fi
fi

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/codex-capture-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  echo "---"
  echo "$ts session=$session_id cwd=$cwd transcript=$transcript_path"
} >> "$log" 2>/dev/null

if [[ "$stop_hook_active" == "true" || "$stop_hook_active" == "1" ]]; then
  echo "$ts skipped: stop hook already active" >> "$log" 2>/dev/null
  exit 0
fi

if [[ -z "$transcript_path" || ! -f "$transcript_path" ]]; then
  echo "$ts skipped: no transcript found" >> "$log" 2>/dev/null
  exit 0
fi

cli_cache="$HOME/.iai-mcp/.cli-path"
iai_cli=""
if [[ -f "$cli_cache" ]]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [[ -x "$cached" ]] && iai_cli="$cached"
fi
if [[ -z "$iai_cli" ]]; then
  path_cli="$(command -v iai-mcp 2>/dev/null || true)"
  if [[ -n "$path_cli" && -x "$path_cli" ]]; then
    iai_cli="$path_cli"
  else
    for candidate in \
      "$HOME/.local/bin/iai-mcp" \
      "$HOME/iai-mcp/.venv/bin/iai-mcp" \
      "$HOME/IAI-MCP/.venv/bin/iai-mcp" \
      "/usr/local/bin/iai-mcp" \
      "/opt/homebrew/bin/iai-mcp"; do
      if [[ -x "$candidate" ]]; then
        iai_cli="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$iai_cli" ]]; then
    printf '%s' "$iai_cli" > "$cli_cache" 2>/dev/null || true
  fi
fi

if [[ -z "$iai_cli" ]]; then
  echo "$ts skipped: iai-mcp CLI not found" >> "$log" 2>/dev/null
  exit 0
fi

if command -v timeout >/dev/null 2>&1; then
  result=$(timeout 30 "$iai_cli" capture-transcript --no-spawn \
    --session-id "${session_id:-codex}" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
elif command -v gtimeout >/dev/null 2>&1; then
  result=$(gtimeout 30 "$iai_cli" capture-transcript --no-spawn \
    --session-id "${session_id:-codex}" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
else
  result=$("$iai_cli" capture-transcript --no-spawn \
    --session-id "${session_id:-codex}" \
    --max-turns 200 \
    "$transcript_path" 2>&1)
fi
rc=$?

{
  echo "$ts rc=$rc result=$result"
} >> "$log" 2>/dev/null

exit 0
