#!/usr/bin/env bash
set -euo pipefail

# Bruteforce Wi-Fi reset for MediaTek MT7925E (mt7925e / mt76)
# Escalation order:
#  1) rfkill unblock + nmcli radio toggle
#  2) restart NetworkManager / wpa_supplicant
#  3) reload mt76 module stack
#  4) locate PCI function (multiple heuristics)
#  5) driver unbind/bind (sysfs)
#  6) function-level PCI reset (if available)
#  7) PCI hot-remove + rescan
#  8) bus-level rescan + settle
#
# If the Wi-Fi device is not enumerated (no PCI function), script cannot recover it without power-cycle.
#
#
# try
#
#
# lspci -nn
# lspci -Dnns | grep -iE "0280|14c3|network|wireless" || true
# sudo dmesg | grep -iE "mt76|mt7925|firmware|pcie|D3|aspm|error|fail" | tail -n 200

DRIVER="mt7925e"
MODULES=(mt7925e mt792x_lib mt76_connac_lib mt76)

log() { printf "[%s] %s\n" "$(date +'%F %T')" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

need_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
}

restart_services() {
  if have systemctl; then
    if systemctl list-unit-files 2>/dev/null | grep -q '^NetworkManager\.service'; then
      log "Restarting NetworkManager"
      systemctl restart NetworkManager || true
    fi
    if systemctl list-unit-files 2>/dev/null | grep -q '^wpa_supplicant\.service'; then
      log "Restarting wpa_supplicant"
      systemctl restart wpa_supplicant || true
    fi
  fi
}

rfkill_nm_toggle() {
  if have rfkill; then
    log "rfkill unblock all (best-effort)"
    rfkill unblock all || true
  fi
  if have nmcli; then
    log "nmcli radio wifi off/on"
    nmcli radio wifi off || true
    sleep 2
    nmcli radio wifi on || true
  fi
}

reload_modules() {
  have modprobe || { log "modprobe missing; skipping module reload"; return 0; }

  # Stop NM to reduce "in use" unload failures
  if have systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^NetworkManager\.service'; then
    systemctl stop NetworkManager || true
  fi

  log "Unloading modules (best-effort): ${MODULES[*]}"
  for m in "${MODULES[@]}"; do
    modprobe -r "$m" 2>/dev/null || true
  done

  log "Reloading modules"
  modprobe mt76 || true
  modprobe mt76_connac_lib || true
  modprobe mt792x_lib || true
  modprobe mt7925e || true

  if have systemctl && systemctl list-unit-files 2>/dev/null | grep -q '^NetworkManager\.service'; then
    systemctl start NetworkManager || true
  fi
}

# Try to find the PCI BDF using multiple strategies.
# Returns: short BDF like "03:00.0" or empty.
find_wifi_bdf() {
  have lspci || { echo ""; return 0; }

  # Strategy A: vendor/class (MediaTek vendor commonly 14c3, wifi class 0280)
  local bdf
  bdf="$(lspci -Dnns 2>/dev/null | awk '
    BEGIN{IGNORECASE=1}
    $2 ~ /^0280:/ && $3 ~ /^14c3:/ {print $1; exit}
  ')"
  [[ -n "$bdf" ]] && { echo "$bdf"; return 0; }

  # Strategy B: any MediaTek "Network controller"
  bdf="$(lspci -Dnns 2>/dev/null | awk '
    BEGIN{IGNORECASE=1}
    /network controller/ && /mediatek/ {print $1; exit}
  ')"
  [[ -n "$bdf" ]] && { echo "$bdf"; return 0; }

  # Strategy C: look for the exact driver name in lspci -k output
  bdf="$(lspci -nnk 2>/dev/null | awk '
    BEGIN{IGNORECASE=1}
    /^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]/ {cur=$1}
    /Kernel driver in use: mt7925e/ {print cur; exit}
  ')"
  [[ -n "$bdf" ]] && { echo "$bdf"; return 0; }

  echo ""
}

full_bdf() { [[ -n "${1:-}" ]] && echo "0000:$1" || echo ""; }

pci_unbind_bind() {
  local bdf_full="$1"
  local dev="/sys/bus/pci/devices/${bdf_full}"
  [[ -d "$dev" ]] || return 1

  if [[ -L "${dev}/driver" ]]; then
    local drv; drv="$(basename "$(readlink -f "${dev}/driver")")"
    log "PCI unbind: ${bdf_full} from ${drv}"
    echo "${bdf_full}" > "${dev}/driver/unbind" || true
    sleep 1
    log "PCI bind: ${bdf_full} to ${drv}"
    echo "${bdf_full}" > "/sys/bus/pci/drivers/${drv}/bind" || true
    return 0
  fi

  # Try binding to mt7925e explicitly if driver exists
  if [[ -d "/sys/bus/pci/drivers/${DRIVER}" ]]; then
    log "PCI bind: ${bdf_full} to ${DRIVER}"
    echo "${bdf_full}" > "/sys/bus/pci/drivers/${DRIVER}/bind" || true
    return 0
  fi

  return 1
}

pci_function_reset() {
  local bdf_full="$1"
  local dev="/sys/bus/pci/devices/${bdf_full}"
  [[ -d "$dev" ]] || return 1

  # Some kernels expose a function-level reset file.
  if [[ -w "${dev}/reset" ]]; then
    log "PCI function reset: ${bdf_full}"
    echo 1 > "${dev}/reset" || true
    return 0
  fi
  return 1
}

pci_runtime_pm_toggle() {
  local bdf_full="$1"
  local dev="/sys/bus/pci/devices/${bdf_full}"
  [[ -d "$dev" ]] || return 1

  local ctrl="${dev}/power/control"
  if [[ -w "$ctrl" ]]; then
    log "Runtime PM: set power/control=on for ${bdf_full}"
    echo on > "$ctrl" || true
    sleep 1
  fi
  return 0
}

pci_remove_rescan() {
  local bdf_full="$1"
  local dev="/sys/bus/pci/devices/${bdf_full}"
  [[ -d "$dev" ]] || return 1

  log "PCI hot-remove: ${bdf_full}"
  echo 1 > "${dev}/remove" || true
  sleep 3
  log "PCI bus rescan"
  echo 1 > /sys/bus/pci/rescan || true
  return 0
}

bus_rescan_and_settle() {
  if [[ -w /sys/bus/pci/rescan ]]; then
    log "Bus-level PCI rescan"
    echo 1 > /sys/bus/pci/rescan || true
  fi
  if have udevadm; then
    log "udevadm settle"
    udevadm settle || true
  fi
}

show_status() {
  log "Interfaces in /sys/class/net:"
  ls -1 /sys/class/net || true
  if have ip; then
    log "ip -brief link:"
    ip -brief link || true
  fi
  if have nmcli; then
    log "nmcli device status:"
    nmcli device status || true
  fi
}

main() {
  need_root
  log "=== MT7925E bruteforce reset (no reboot) ==="

  rfkill_nm_toggle
  restart_services
  reload_modules
  rfkill_nm_toggle
  restart_services

  local short; short="$(find_wifi_bdf || true)"
  if [[ -z "$short" ]]; then
    log "FAIL: No Wi-Fi PCI function detected via lspci."
    log "This typically means the device is not enumerated (PCIe/firmware/power-state wedge)."
    log "Non-reboot recovery is usually not possible in this state."
    show_status
    exit 2
  fi

  local bdf; bdf="$(full_bdf "$short")"
  log "Found Wi-Fi PCI device: ${bdf}"

  pci_runtime_pm_toggle "$bdf" || true
  pci_unbind_bind "$bdf" || true
  sleep 2
  reload_modules
  rfkill_nm_toggle
  restart_services

  pci_function_reset "$bdf" || true
  sleep 2
  reload_modules
  rfkill_nm_toggle
  restart_services

  pci_remove_rescan "$bdf" || true
  sleep 2
  bus_rescan_and_settle
  reload_modules
  rfkill_nm_toggle
  restart_services

  show_status
  log "Done."
}

main "$@"
