
frame=__AI_SDK_WRITE_PROTECTION_FRAME__
b64() { printf '%s' "$1" | base64 | tr -d '\r\n'; }
emit() {
  printf '%s' "$1"
  shift
  for emit_field in "$@"; do printf ' %s' "$emit_field"; done
  printf '\n'
}
canonical() {
  canonical_framed=$(realpath "$1" 2>/dev/null; canonical_status=$?; printf '%s' "$frame"; exit "$canonical_status") || return 1
  canonical_path=${canonical_framed%"$frame"}
  canonical_path=${canonical_path%$'\n'}
  case "$canonical_path" in /*) ;; *) return 1 ;; esac
}
resolve() {
  resolve_target=$1
  resolve_missing=
  while [ ! -e "$resolve_target" ] && [ ! -L "$resolve_target" ]; do
    resolve_parent=${resolve_target%/*}
    [ -n "$resolve_parent" ] || resolve_parent=/
    resolve_segment=${resolve_target##*/}
    if [ "$resolve_parent" = "$resolve_target" ] || [ -z "$resolve_segment" ] || [ "$resolve_segment" = . ] || [ "$resolve_segment" = .. ]; then
      return 1
    fi
    resolve_missing=/$resolve_segment$resolve_missing
    resolve_target=$resolve_parent
  done
  if [ -L "$resolve_target" ] && [ ! -e "$resolve_target" ]; then return 1; fi
  canonical "$resolve_target" || return 1
  resolved=${canonical_path%/}$resolve_missing
  [ -n "$resolved" ] || resolved=/
}
scan() {
  (
    set -o pipefail
    find "$1" -mindepth 1 -print0 | {
      scan_count=0
      while IFS= read -r -d '' scan_entry; do
        scan_count=$((scan_count + 1))
        if [ "$scan_count" -gt "$scan_budget" ]; then
          emit X
          exit 0
        fi
        if [ -L "$scan_entry" ]; then
          if [ ! -e "$scan_entry" ] || ! canonical "$scan_entry"; then
            exit 1
          fi
          if [ -d "$canonical_path" ]; then scan_is_dir=1; else scan_is_dir=0; fi
          emit S "$scan_is_dir" "$(b64 "$canonical_path")"
        fi
      done
      emit C "$scan_count"
    }
  )
}
