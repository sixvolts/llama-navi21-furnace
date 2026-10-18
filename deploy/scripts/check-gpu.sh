#!/bin/bash
# Check that a Radeon PRO V620 is fully usable: kernel parameters, the 32 GB BAR, ECC state,
# the compute (KFD) node, and ROCm. No root needed. Run after every BIOS or kernel change.
#
#   deploy/scripts/check-gpu.sh [path/to/llama.cpp/build]
#
# The optional build directory adds a llama.cpp device listing at the end.

build=${1:-}
ok=1
fail() { echo "  FAIL: $*"; ok=0; }

echo "== kernel parameters =="
tr ' ' '\n' < /proc/cmdline | grep -E '^(pci=|amdgpu\.)' | sed 's/^/  /' || echo "  (none)"
grep -qw 'amdgpu.ras_enable=0' /proc/cmdline || echo "  note: amdgpu.ras_enable=0 is not set - VRAM ECC stays on (about -11% decode speed)"

echo
echo "== PCI: AMD GPUs and their memory BARs =="
cards=()
for dev in /sys/bus/pci/devices/*; do
  [ "$(cat $dev/vendor)" = 0x1002 ] || continue
  case "$(cat $dev/class)" in 0x030000|0x038000|0x120000) ;; *) continue;; esac
  addr=$(basename $dev)
  drv=$(basename "$(readlink $dev/driver 2>/dev/null)" 2>/dev/null)
  name=$(lspci -s $addr 2>/dev/null | cut -d: -f3- | sed 's/^ //')
  echo "  $addr  ${name:-?}  driver=${drv:-none}  link=$(cat $dev/current_link_speed 2>/dev/null | cut -d' ' -f1-2) x$(cat $dev/current_link_width 2>/dev/null)"
  # largest BAR from the resource file (start end flags per line)
  big=$(awk '{s=strtonum($1); e=strtonum($2); if (e>s && e-s+1>m) m=e-s+1} END {printf "%d", m/1048576}' $dev/resource)
  echo "      largest BAR: ${big} MiB"
  if echo "$name" | grep -qi 'V620'; then
    cards+=($addr)
    [ "${big:-0}" -ge 32768 ] || fail "$addr: the 32 GB VRAM BAR is not assigned - enable Above 4G Decoding and disable CSM in the BIOS"
  fi
done
[ ${#cards[@]} -gt 0 ] || fail "no Radeon PRO V620 found on the PCI bus"

echo
echo "== above-4G memory window (needed for the 32 GB BAR) =="
if journalctl -k -b 0 --no-pager 2>/dev/null | grep -q 'root bus resource'; then
  journalctl -k -b 0 --no-pager | grep 'root bus resource \[mem' | grep -vE '\[mem 0x0*[0-9a-f]{1,8}-0x0*[0-9a-f]{1,8} ' | sed 's/.*\(root bus resource.*\)/  \1/' | head -3
else
  echo "  (kernel log not readable - add yourself to the adm group, or run with sudo)"
fi

echo
echo "== amdgpu: VRAM size and ECC =="
if journalctl -k -b 0 --no-pager >/dev/null 2>&1; then
  journalctl -k -b 0 --no-pager | grep -E 'Detected VRAM RAM|GECC' | sed 's/.*kernel: /  /' | sort -u
  if journalctl -k -b 0 --no-pager | grep -q 'GECC will be enabled in next boot'; then
    fail "ECC turns back ON at the next boot - this boot ran without amdgpu.ras_enable=0 (section 4)"
  elif journalctl -k -b 0 --no-pager | grep -q 'GECC will be disabled in next boot'; then
    echo "  note: ECC turns off on the NEXT boot - reboot once more"
  elif ! journalctl -k -b 0 --no-pager | grep -q 'GECC is disabled'; then
    fail "VRAM ECC is on - set amdgpu.ras_enable=0 and reboot twice (section 4)"
  fi
fi

echo
echo "== KFD compute nodes (a V620 shows simd_count 144) =="
found=0
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
  sc=$(awk '/^simd_count/ {print $2}' $n/properties 2>/dev/null)
  gfx=$(awk '/^gfx_target_version/ {print $2}' $n/properties 2>/dev/null)
  echo "  node $(basename $n): simd_count=${sc:-?} gfx_target_version=${gfx:-?}"
  [ "${sc:-0}" -gt 0 ] && found=1
done
[ $found = 1 ] || fail "no GPU compute node - amdgpu did not initialise the card (check dmesg)"

echo
echo "== ROCm =="
if command -v rocminfo >/dev/null; then
  timeout 30 rocminfo 2>&1 | grep -E 'Marketing Name|^\s+Name:\s+gfx' | sed 's/^ */  /' | head -6
  timeout 30 rocminfo >/dev/null 2>&1 || fail "rocminfo failed - is your user in the render and video groups?"
else
  fail "rocminfo not installed"
fi

if [ -n "$build" ] && [ -x "$build/bin/llama-cli" ]; then
  echo
  echo "== llama.cpp devices =="
  timeout 60 "$build/bin/llama-cli" --list-devices 2>&1 | grep -E 'ROCm|Device' | head -4 | sed 's/^/  /'
fi

echo
[ $ok = 1 ] && echo "RESULT: OK" || { echo "RESULT: problems found (see FAIL lines)"; exit 1; }
