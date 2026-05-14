#!/usr/bin/env bash
# IAI-MCP SessionStart hook — recall injection.
#
# Fires on Claude Code session start (sources: startup, resume, clear,
# compact). Reads the stdin JSON for session_id and source, invokes the
# iai-mcp CLI to fetch the cached session prefix from the daemon, and prints
# the result to stdout for Claude Code to inject as additionalContext. The
# CLI itself caps stdout at 10000 characters; this script relays the bytes
# verbatim.
#
# Fail-safe by design: every error path exits 0 with empty stdout so a
# recall miss never blocks session start. Logs go to
# ~/.iai-mcp/logs/recall-YYYY-MM-DD.log for audit.

set -u  # no -e: fail-safe is paramount
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
source_evt=$(extract "source")

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/recall-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
  echo "---"
  echo "$ts session=$session_id source=$source_evt"
} >> "$log" 2>/dev/null

# Read the daemon-written cache whenever it is non-empty (no age cap).
# Each branch writes a contract log marker. Falls through to the live CLI path
# on any miss.
cache_path="$HOME/.iai-mcp/.session-start-payload.cached.md"
if [[ -s "$cache_path" ]]; then
  # Cross-platform mtime: try GNU stat, then BSD stat.
  cache_mtime=$(stat -c %Y "$cache_path" 2>/dev/null || stat -f %m "$cache_path" 2>/dev/null || echo 0)
  if [[ $cache_mtime -eq 0 ]]; then
    echo "$ts cache-error stat-failed" >> "$log" 2>/dev/null
  else
    now_epoch=$(date +%s)
    age=$(( now_epoch - cache_mtime ))
    cache_out=$(head -c 10000 "$cache_path" 2>/dev/null || true)
    if [[ -n "$cache_out" ]]; then
      printf '%s' "$cache_out"
      echo "$ts cache-hit age=${age}s bytes=${#cache_out}" >> "$log" 2>/dev/null
      exit 0
    fi
    echo "$ts cache-miss empty (file existed but read returned 0 bytes)" >> "$log" 2>/dev/null
  fi
elif [[ -e "$cache_path" ]]; then
  echo "$ts cache-miss empty (zero-byte file)" >> "$log" 2>/dev/null
else
  echo "$ts cache-miss absent" >> "$log" 2>/dev/null
fi

# Locate the CLI. Lookup order:
#   1. IAI_MCP_SESSION_RECALL_CLI environment variable (developer override
#      for non-standard install locations; export in your shell init).
#   2. ~/.iai-mcp/.cli-path cache file (auto-populated below once the
#      candidates array finds a working binary).
#   3. Generic install locations in the candidates array.
# Only generic install paths are baked into the source; developer-specific
# paths belong in the env var or the cache, never here.
cli_cache="$HOME/.iai-mcp/.cli-path"
iai_cli=""
if [[ -n "${IAI_MCP_SESSION_RECALL_CLI:-}" && -x "$IAI_MCP_SESSION_RECALL_CLI" ]]; then
  iai_cli="$IAI_MCP_SESSION_RECALL_CLI"
fi
if [[ -z "$iai_cli" && -f "$cli_cache" ]]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [[ -x "$cached" ]] && iai_cli="$cached"
fi
if [[ -z "$iai_cli" ]]; then
  candidates=(
    "$HOME/IAI-MCP/.venv/bin/iai-mcp"
    "/usr/local/bin/iai-mcp"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      iai_cli="$candidate"
      printf '%s' "$iai_cli" > "$cli_cache" 2>/dev/null || true
      break
    fi
  done
fi
if [[ -z "$iai_cli" ]]; then
  echo "$ts skipped: iai-mcp CLI not found" >> "$log" 2>/dev/null
  exit 0
fi

# Hard cap on the CLI call. Default 10s; IAI_MCP_RECALL_HOOK_TIMEOUT overrides
# the cap (used by failsafe contract tests to cap at 2s against sleeping
# stubs). On cap-exceed the CLI yields no stdout, not a hang.
hook_timeout="${IAI_MCP_RECALL_HOOK_TIMEOUT:-10}"
if command -v timeout >/dev/null 2>&1; then
  out=$(timeout "$hook_timeout" "$iai_cli" session-start --session-id "$session_id" 2>>"$log")
  rc=$?
elif command -v gtimeout >/dev/null 2>&1; then
  out=$(gtimeout "$hook_timeout" "$iai_cli" session-start --session-id "$session_id" 2>>"$log")
  rc=$?
else
  # Pure-bash watchdog when coreutils is absent: launch CLI in background,
  # capture stdout via a temp file, kill on cap-exceed. Keeps the hook
  # fail-safe on minimal POSIX systems.
  tmp_out=$(mktemp 2>/dev/null || echo "/tmp/iai-mcp-recall-$$.out")
  "$iai_cli" session-start --session-id "$session_id" >"$tmp_out" 2>>"$log" &
  cli_pid=$!
  killed=0
  for ((i=0; i<hook_timeout*10; i++)); do
    if ! kill -0 "$cli_pid" 2>/dev/null; then break; fi
    sleep 0.1
  done
  if kill -0 "$cli_pid" 2>/dev/null; then
    kill -TERM "$cli_pid" 2>/dev/null
    sleep 0.2
    kill -KILL "$cli_pid" 2>/dev/null
    killed=1
  fi
  wait "$cli_pid" 2>/dev/null
  rc=$?
  if [[ $killed -eq 1 ]]; then
    rc=124
    out=""
  else
    out=$(cat "$tmp_out" 2>/dev/null || true)
  fi
  rm -f "$tmp_out" 2>/dev/null || true
fi

if [[ $rc -eq 0 ]]; then
  printf '%s' "$out"
fi
{
  echo "$ts rc=$rc bytes=${#out}"
} >> "$log" 2>/dev/null
exit 0
