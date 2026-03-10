#!/usr/bin/env bash
## @file ralph_wiggum.sh
## @brief Runs Claude Code in a loop against an implementation plan, requiring
##        each pass to complete the next actionable item, update the plan, run
##        focused tests, perform review, and commit its work.
##
## Supports three modes:
##   plan  — Refine a plan file by reading source files and expanding high-level
##           items into concrete, verifiable sub-items. No code changes.
##   work  — Execute the next actionable item, test, review, commit. (default)
##   full  — Run plan refinement first, then switch to work mode.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ralph_wiggum.sh [options] [plan-file]

Runs Claude Code non-interactively in a Ralph Wiggum loop against one
implementation plan file. Each work pass is expected to:
  - read the plan and the active phase's "Read first" files,
  - do the next actionable item,
  - split oversized items into smaller verifiable pieces when needed,
  - run focused tests,
  - update the plan in place,
  - commit its work,
  - and go through an automated review pass that may update the plan.

Options:
  --plan FILE               Implementation plan file to drive.
  --mode MODE               Execution mode: plan, work, or full (default: work).
  --max-iterations N        Stop after N outer iterations per mode (0 = unlimited).
  --model MODEL             Claude model override.
  --log-dir DIR             Directory for prompts, outputs, and logs.
  --allow-dirty             Allow starting from a dirty worktree.
  -h, --help                Show this help text.

Modes:
  plan    Read source files listed in the plan's "Read first" section and expand
          high-level items into concrete, verifiable sub-items. Each iteration
          refines one section. No code changes are made — only the plan file is
          updated and committed.
  work    (default) Execute the next actionable item, run tests, review, commit.
  full    Run plan refinement iterations first, then switch to work iterations.
          --max-iterations applies independently to each phase.

Environment overrides:
  RALPH_WIGGUM_PLAN
  RALPH_WIGGUM_MODE
  RALPH_WIGGUM_MAX_ITERATIONS
  RALPH_WIGGUM_MODEL
  RALPH_WIGGUM_LOG_DIR
  RALPH_WIGGUM_ALLOW_DIRTY=1
  RALPH_WIGGUM_CLAUDE_FLAGS='...'

Examples:
  ralph_wiggum.sh --plan plans/00-foundation.md --mode plan --max-iterations 5
  ralph_wiggum.sh --plan plans/00-foundation.md --mode work --max-iterations 10
  ralph_wiggum.sh --plan plans/00-foundation.md --mode full --max-iterations 5
  ralph_wiggum.sh --max-iterations 3 doc/hypre-semi-struct-plan.md
USAGE
}

die() {
  printf 'ralph_wiggum: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

abs_path() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$PWD" "$path"
  fi
}

git_status_snapshot() {
  git status --porcelain --untracked-files=all
}

assert_status_matches_baseline() {
  local context=$1
  local current
  current=$(git_status_snapshot)
  if [[ "$current" != "$BASELINE_STATUS" ]]; then
    printf 'ralph_wiggum: worktree drift detected after %s\n' "$context" >&2
    printf '\n--- baseline ---\n%s\n' "$BASELINE_STATUS" >&2
    printf '\n--- current ---\n%s\n' "$current" >&2
    exit 2
  fi
}

# ─── Status line protocol ───────────────────────────────────────────────────
#
# Work and plan passes must end their output with exactly two lines:
#   RALPH_STATUS: committed|done|blocked
#   RALPH_SUMMARY: <one-line description>
#
# The review pass is plain text with no status protocol.

STATUS_INSTRUCTIONS='
When you are finished, you MUST end your response with exactly these two lines
(no extra whitespace, no markdown fencing):

RALPH_STATUS: <status>
RALPH_SUMMARY: <one-line summary>

Where <status> is one of: committed, done, blocked
And <summary> is a concise one-line description of what you did.'

extract_status() {
  local output_file=$1
  grep -m1 '^RALPH_STATUS: ' "$output_file" | sed 's/^RALPH_STATUS: //' | tr -d '[:space:]'
}

extract_summary() {
  local output_file=$1
  grep -m1 '^RALPH_SUMMARY: ' "$output_file" | sed 's/^RALPH_SUMMARY: //'
}

# ─── Prompt builders ─────────────────────────────────────────────────────────

build_plan_prompt() {
  local prompt_file=$1
  cat > "$prompt_file" <<EOF_PROMPT
You are running inside an automated Ralph Wiggum planning loop.

Repository root: $REPO_ROOT
Implementation plan: $PLAN_FILE

Your job is to refine the implementation plan by reading the codebase and
expanding high-level items into concrete, verifiable sub-items. You must NOT
make any code changes — only update the plan file.

Required process:
1. Read the implementation plan first.
2. Read the "Read first" files listed in the plan. These are the actual source
   files you need to understand before refining plan items.
3. If the plan references a meta/decision-log file (e.g. plans/meta.md), read
   it too. Honor any decisions already recorded there.
4. Choose the next section or item that needs refinement. Look for items that
   are too high-level, vague, or large to complete in a single work pass
   (roughly: >300 lines of diff, >3 files touched, or requiring non-obvious
   design decisions).
5. For each item you refine, produce sub-items that specify:
   - The exact file(s) to modify or create.
   - The specific range-v3 patterns to replace and what to replace them with.
   - A concrete test command to verify the change.
   - Any ordering constraints (must come before/after another item).
6. If you discover cross-cutting decisions that affect multiple plan phases,
   record them in the decision log (plans/meta.md) as new TBD decisions.
7. If you encounter items that are already concrete enough, leave them as-is
   and move to the next section.
8. Update the plan file in place with your refinements.
9. Commit the plan update with git. Never use --no-verify.
10. If the plan is fully refined (all items are concrete and verifiable), make
    no changes and return status "done".
11. If you are blocked (e.g., missing information, ambiguous requirements),
    return status "blocked" with a concise explanation.

Important constraints:
- Do NOT modify any source code. Only modify plan files and decision logs.
- Keep each sub-item small enough for one work pass (~1-2 files, <300 lines).
- Preserve existing completed items (marked with [x]) unchanged.
- The plan file is the single source of truth for progress.
${STATUS_INSTRUCTIONS}
EOF_PROMPT
}

build_plan_review_prompt() {
  local prompt_file=$1
  local commit_sha=$2
  cat > "$prompt_file" <<EOF_PROMPT
You are running inside an automated Ralph Wiggum plan-review pass.

Repository root: $REPO_ROOT
Implementation plan: $PLAN_FILE
Review target commit: $commit_sha

Your job is to review the plan refinement and ensure quality.

Required process:
1. Read the implementation plan.
2. Inspect commit $commit_sha (the plan refinement) and assess whether:
   - Sub-items are specific enough (name exact files, patterns, test commands).
   - Sub-items are correctly ordered (dependencies respected).
   - No items were accidentally deleted or corrupted.
   - Cross-cutting decisions are recorded in the decision log if needed.
3. If you find issues, update the plan in place and commit. Never use --no-verify.
4. Do not modify source code.
5. Return a concise plain-text summary of what you reviewed and whether you
   updated the plan.
EOF_PROMPT
}

build_work_prompt() {
  local prompt_file=$1
  cat > "$prompt_file" <<EOF_PROMPT
You are running inside an automated Ralph Wiggum loop.

Repository root: $REPO_ROOT
Implementation plan: $PLAN_FILE

Your job is to read the implementation plan, identify the next incomplete
actionable item, and complete the next useful chunk of work.

Required process:
1. Read the implementation plan first.
2. Read the active phase's "Read first" files before making design or
   implementation decisions.
3. Choose the next incomplete actionable item. Do not skip ahead.
4. If that item is too large for one pass, break it into smaller verifiable
   subitems in the plan file, in the correct phase.  Don't bother trying to complete
   any of the new items.  Properly updating the plan is enough.
5. If parallel exploration or disjoint implementation will help, spawn multiple
   agents and use them.
6. Make only directly related changes. Ignore unrelated files and unrelated
   dirty state.
7. Run focused tests for the work you changed. If you change build or test
   wiring, run the narrowest relevant build/test commands.
8. Review your own work before finishing.
9. Update the implementation plan in place so the next launch can continue from
   the correct location. Record completions, tests, and any new follow-up items
   in the right phase.
10. Commit your work with git. Never use --no-verify.
11. If the plan is complete, make no changes and return status "done".
12. If you are blocked, update the plan if appropriate and return status
    "blocked" with a concise explanation.

Important constraints:
- The plan file is the single source of truth for progress.
- The first response must not ask the human for routine decisions that can be
  resolved from the repo and plan.
- Keep changes small, verifiable, and directly tied to the next plan item.
- Leave the repository in a committed state if you made changes.
${STATUS_INSTRUCTIONS}
EOF_PROMPT
}

build_review_prompt() {
  local prompt_file=$1
  local commit_sha=$2
  cat > "$prompt_file" <<EOF_PROMPT
You are running inside an automated Ralph Wiggum review pass.

Repository root: $REPO_ROOT
Implementation plan: $PLAN_FILE
Review target commit: $commit_sha

Your job is to review the completed work and record any actionable follow-up
items in the implementation plan so the next work pass can address them.

Required process:
1. Read the implementation plan.
2. Inspect commit $commit_sha and the files it changed before making review
   decisions.
3. Focus only on actionable issues related to:
   - correctness,
   - missed or insufficient tests,
   - plan-file consistency,
   - risky omissions in the completed work.
4. Do not suggest optional stylistic cleanups.
5. If you find actionable issues, update the implementation plan in place with
   concrete follow-up items in the correct phase and order so they become the
   next work to do.
6. Commit the plan update if you changed it. Never use --no-verify.
7. Do not fix code directly in this review pass.
8. If there are no actionable issues, make no changes.
9. Return a concise plain-text summary of what you reviewed and whether you
   updated the plan.
EOF_PROMPT
}

# ─── Claude runner ───────────────────────────────────────────────────────────
#
# Claude Code CLI non-interactive usage:
#   claude -p "prompt"              Print response to stdout and exit
#   --dangerously-skip-permissions  Bypass permission checks (container use)
#   --no-session-persistence        Don't save session to disk
#   --model MODEL                   Override model
#
# All output is plain text on stdout.  No JSON envelope, no schema extraction.

run_claude() {
  local prompt_file=$1
  local output_file=$2
  local log_file=$3

  local prompt
  prompt=$(cat "$prompt_file")

  local cmd=(claude -p --dangerously-skip-permissions --no-session-persistence)
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ ${#CLAUDE_EXTRA_FLAGS[@]} -gt 0 ]]; then
    cmd+=("${CLAUDE_EXTRA_FLAGS[@]}")
  fi

  # stdout = claude's response, stderr = diagnostics
  # Unset CLAUDECODE to allow nested invocations (e.g., running from inside
  # an existing Claude Code session in a container).
  env -u CLAUDECODE "${cmd[@]}" "$prompt" 2>"$log_file" | tee "$output_file" || {
    printf 'ralph_wiggum: claude exited with error\n' >&2
    cat "$log_file" >&2
    die "claude pass failed (see $log_file)"
  }

  [[ -s "$output_file" ]] || die "claude produced empty output (see $log_file)"
}

# ─── Pass orchestrators ──────────────────────────────────────────────────────

run_claude_plan() {
  local iteration=$1
  local prefix="$LOG_DIR/iteration-$(printf '%04d' "$iteration")-plan"

  build_plan_prompt "${prefix}.prompt"

  printf '==> claude plan pass: iteration %d\n' "$iteration" >&2
  run_claude "${prefix}.prompt" "${prefix}.out" "${prefix}.log"
  printf '%s\n' "${prefix}.out"
}

run_claude_plan_review() {
  local iteration=$1
  local commit_sha=$2
  local prefix="$LOG_DIR/iteration-$(printf '%04d' "$iteration")-plan-review"

  build_plan_review_prompt "${prefix}.prompt" "$commit_sha"

  printf '==> claude plan review: iteration %d\n' "$iteration" >&2
  run_claude "${prefix}.prompt" "${prefix}.out" "${prefix}.log"
  printf '%s\n' "${prefix}.out"
}

run_claude_work() {
  local iteration=$1
  local prefix="$LOG_DIR/iteration-$(printf '%04d' "$iteration")-work"

  build_work_prompt "${prefix}.prompt"

  printf '==> claude work pass: iteration %d\n' "$iteration" >&2
  run_claude "${prefix}.prompt" "${prefix}.out" "${prefix}.log"
  printf '%s\n' "${prefix}.out"
}

run_claude_review() {
  local iteration=$1
  local commit_sha=$2
  local prefix="$LOG_DIR/iteration-$(printf '%04d' "$iteration")-review"

  build_review_prompt "${prefix}.prompt" "$commit_sha"

  printf '==> claude review: iteration %d\n' "$iteration" >&2
  run_claude "${prefix}.prompt" "${prefix}.out" "${prefix}.log"
  printf '%s\n' "${prefix}.out"
}

# ─── Main loops ──────────────────────────────────────────────────────────────

run_plan_loop() {
  local max_iter=$1
  local iteration=1
  local last_head
  last_head=$(git rev-parse HEAD)

  printf 'ralph_wiggum: starting plan refinement loop (max_iterations=%s)\n' "$max_iter" >&2

  while :; do
    if [[ "$max_iter" != 0 && $iteration -gt $max_iter ]]; then
      printf 'ralph_wiggum: plan loop reached max iterations (%s)\n' "$max_iter" >&2
      break
    fi

    local plan_output plan_status plan_summary
    plan_output=$(run_claude_plan "$iteration")
    plan_status=$(extract_status "$plan_output")
    plan_summary=$(extract_summary "$plan_output")

    printf '==> plan result [%s]: %s\n' "$plan_status" "$plan_summary" >&2

    if [[ -z "$plan_status" ]]; then
      printf 'ralph_wiggum: could not extract RALPH_STATUS from output\n' >&2
      printf '==> last 10 lines of output:\n' >&2
      tail -10 "$plan_output" >&2
      die "status extraction failed"
    fi

    case "$plan_status" in
      done)
        assert_status_matches_baseline "plan done status"
        printf 'ralph_wiggum: plan is fully refined\n' >&2
        break
        ;;
      blocked)
        assert_status_matches_baseline "plan blocked status"
        die "plan blocked: $plan_summary"
        ;;
      committed)
        current_head=$(git rev-parse HEAD)
        [[ "$current_head" != "$last_head" ]] || die 'Claude reported a plan commit, but HEAD did not advance'
        # Update baseline: the commit legitimately changed tracked-file state
        BASELINE_STATUS=$(git_status_snapshot)
        ;;
      *)
        die "unexpected plan status: $plan_status"
        ;;
    esac

    local review_output review_summary
    review_output=$(run_claude_plan_review "$iteration" "$current_head")
    review_summary=$(cat "$review_output")

    printf '==> plan review summary:\n%s\n' "$review_summary" >&2
    # Review may have committed plan updates — refresh baseline
    BASELINE_STATUS=$(git_status_snapshot)

    last_head=$(git rev-parse HEAD)
    iteration=$((iteration + 1))
  done
}

run_work_loop() {
  local max_iter=$1
  local iteration=1
  local last_head
  last_head=$(git rev-parse HEAD)

  printf 'ralph_wiggum: starting work loop (max_iterations=%s)\n' "$max_iter" >&2

  while :; do
    if [[ "$max_iter" != 0 && $iteration -gt $max_iter ]]; then
      printf 'ralph_wiggum: work loop reached max iterations (%s)\n' "$max_iter" >&2
      break
    fi

    local work_output work_status work_summary
    work_output=$(run_claude_work "$iteration")
    work_status=$(extract_status "$work_output")
    work_summary=$(extract_summary "$work_output")

    printf '==> work result [%s]: %s\n' "$work_status" "$work_summary" >&2

    if [[ -z "$work_status" ]]; then
      printf 'ralph_wiggum: could not extract RALPH_STATUS from output\n' >&2
      printf '==> last 10 lines of output:\n' >&2
      tail -10 "$work_output" >&2
      die "status extraction failed"
    fi

    case "$work_status" in
      done)
        assert_status_matches_baseline "done status"
        printf 'ralph_wiggum: plan reports no more actionable work\n' >&2
        break
        ;;
      blocked)
        assert_status_matches_baseline "blocked status"
        die "blocked: $work_summary"
        ;;
      committed)
        current_head=$(git rev-parse HEAD)
        [[ "$current_head" != "$last_head" ]] || die 'Claude reported a commit, but HEAD did not advance'
        # Update baseline: the commit legitimately changed tracked-file state
        BASELINE_STATUS=$(git_status_snapshot)
        ;;
      *)
        die "unexpected work status: $work_status"
        ;;
    esac

    local review_output review_summary
    review_output=$(run_claude_review "$iteration" "$current_head")
    review_summary=$(cat "$review_output")

    printf '==> review summary:\n%s\n' "$review_summary" >&2
    # Review may have committed plan updates — refresh baseline
    BASELINE_STATUS=$(git_status_snapshot)

    last_head=$(git rev-parse HEAD)
    iteration=$((iteration + 1))
  done
}

# ─── Entry point ─────────────────────────────────────────────────────────────

main() {
  PLAN_FILE=${RALPH_WIGGUM_PLAN:-doc/hypre-semi-struct-plan.md}
  MAX_ITERATIONS=${RALPH_WIGGUM_MAX_ITERATIONS:-0}
  MODE=${RALPH_WIGGUM_MODE:-work}
  MODEL=${RALPH_WIGGUM_MODEL:-}
  ALLOW_DIRTY=${RALPH_WIGGUM_ALLOW_DIRTY:-0}
  LOG_DIR=${RALPH_WIGGUM_LOG_DIR:-}

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --plan)
        PLAN_FILE=$2
        shift 2
        ;;
      --mode)
        MODE=$2
        shift 2
        ;;
      --max-iterations)
        MAX_ITERATIONS=$2
        shift 2
        ;;
      --model)
        MODEL=$2
        shift 2
        ;;
      --log-dir)
        LOG_DIR=$2
        shift 2
        ;;
      --allow-dirty)
        ALLOW_DIRTY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        die "unknown option: $1"
        ;;
      *)
        PLAN_FILE=$1
        shift
        ;;
    esac
  done

  case "$MODE" in
    plan|work|full) ;;
    *) die "unknown mode: $MODE (must be plan, work, or full)" ;;
  esac

  require_cmd claude
  require_cmd git

  REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
  [[ -n "$REPO_ROOT" ]] || die 'must run inside a git repository'
  cd "$REPO_ROOT"

  PLAN_FILE=$(abs_path "$PLAN_FILE")
  [[ -f "$PLAN_FILE" ]] || die "plan file not found: $PLAN_FILE"

  BASELINE_STATUS=$(git_status_snapshot)
  if [[ "$ALLOW_DIRTY" != 1 && -n "$BASELINE_STATUS" ]]; then
    printf 'ralph_wiggum: refusing to start from a dirty worktree\n' >&2
    printf '%s\n' "$BASELINE_STATUS" >&2
    printf 'Use --allow-dirty or RALPH_WIGGUM_ALLOW_DIRTY=1 to override.\n' >&2
    exit 2
  fi

  if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$REPO_ROOT/.git/ralph-wiggum"
  fi
  mkdir -p "$LOG_DIR"

  CLAUDE_EXTRA_FLAGS=()
  if [[ -n "${RALPH_WIGGUM_CLAUDE_FLAGS:-}" ]]; then
    read -r -a CLAUDE_EXTRA_FLAGS <<<"$RALPH_WIGGUM_CLAUDE_FLAGS"
  fi

  printf 'ralph_wiggum: mode=%s plan=%s max_iterations=%s\n' "$MODE" "$PLAN_FILE" "$MAX_ITERATIONS" >&2

  case "$MODE" in
    plan)
      run_plan_loop "$MAX_ITERATIONS"
      ;;
    work)
      run_work_loop "$MAX_ITERATIONS"
      ;;
    full)
      run_plan_loop "$MAX_ITERATIONS"
      run_work_loop "$MAX_ITERATIONS"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
