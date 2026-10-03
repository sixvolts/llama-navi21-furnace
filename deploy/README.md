# Deploying on one Radeon PRO V620

A linux machine with a single AMD Radeon PRO V620 running a 27B model at 40-75 tokens/s,
with web search tools, from bare hardware: BIOS, OS, kernel parameters, ROCm, llama.cpp, models,
and the server.

The V620 is a 32 GB Navi 21 (gfx1030, RDNA2) datacenter card. It sells cheaply second-hand and has
the memory for a 27B model at 4 bits with 128k context, but it needs a few BIOS and kernel settings
that a gaming card does not, and ROCm's defaults leave speed on the table. Everything below is
what made it work.

This describes a working machine as of 2026-10-03. It has not yet been replayed from a clean
install; please open an issue for anything that differs.

**What you end up with**

- [Swift-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) (UkisAI's fine-tune of
  Qwen3.8-27B) at Q4_K_XL, 128k context, fully on the GPU (about 28 GB of VRAM with the
  DFlash2 drafter at full context)
- speculative decoding with a DFlash2 or MTP drafter, about 43 t/s on research-style chat and up
  to 75 t/s on math, against 24 t/s without
- llama-server's web UI with the web search tools, system prompt and defaults from
  [llama-webui-tools](https://github.com/sixvolts/llama-webui-tools)

| | |
|---|---|
| 1. [Hardware](#1-hardware) | 7. [Build llama.cpp](#7-build-llamacpp) |
| 2. [BIOS](#2-bios) | 8. [Models](#8-models) |
| 3. [Operating system](#3-operating-system) | 9. [Web tools and prompt](#9-web-tools-and-prompt) |
| 4. [Kernel parameters](#4-kernel-parameters) | 10. [Run the server](#10-run-the-server) |
| 5. [ROCm and build tools](#5-rocm-and-build-tools) | 11. [Start at boot](#11-start-at-boot) |
| 6. [Check the GPU](#6-check-the-gpu) | 12. [Troubleshooting](#12-troubleshooting) |

**Files in this directory**

```
README.md                    this guide
serve-v620.sh                llama-server launcher (model, drafter, web tools, prompt, UI defaults)
scripts/install-packages.sh  Ubuntu packages, user groups, pinned cmake
scripts/check-gpu.sh         checks BAR, ECC, compute node and ROCm; run after any change
scripts/build-llama.sh       builds this checkout for gfx1030
scripts/download-models.sh   downloads the model and drafters from Hugging Face
scripts/make-models.sh       or rebuilds them from the upstream sources
recipes/swift-q4_k_xl.txt    per-tensor quantization types used by make-models.sh
systemd/llama-server.service systemd user service
power/                       V620 power-cap patch for amdgpu, boot script and unit (section 4)
huggingface/                 model card, licenses and upload script for the model files
```

---

## 1. Hardware

Tested on: Gigabyte X570 AORUS PRO WIFI (BIOS F40c), Ryzen 7 5800XT, 32 GB RAM, one V620, and a
Radeon PRO WX 3200 for the display.

- **The V620 has no display outputs.** You need another display source for the BIOS setup and
  the console: a CPU with integrated graphics, or a second inexpensive GPU. Any card works; ROCm
  does not need to support it.
- **Power:** 300 W through two 8-pin PCIe connectors; AMD recommends a 700 W power supply.
- **Slot:** PCIe 4.0 x16, dual slot, 267 mm long. Use a CPU-connected x16 slot.
- **Cooling:** it is built for a server chassis. Check whether your card has its own fan; if it
  does not, it needs air pushed through it, or it will throttle and shut down. At the 300 W cap
  (section 4) a sustained long prompt is the hard case: the test machine (a 97 mm blower on a
  printed shroud plus two 120 mm fans along the card, cool intake air, shroud sealed at the
  power connectors) settles at a junction of 98 to 99 C against a 100 C limit and loses about
  1% to the clock trim over an 8-minute prefill; with warm intake air it sat at 100 to 102 C
  and lost 5 to 8%. The junction sits 22 to 25 C above the edge sensor at this power, which is
  normal for the die-to-heatsink interface. Short prompts (under about 40k tokens) never reach
  the limit. Watch `temp2_input` (junction) in the card's hwmon directory.
- **RAM:** 32 GB is enough; the model lives on the GPU. Disk: about 20 GB for the model files
  (about 80 GB more to rebuild them yourself, section 8).

## 2. BIOS

Setting names below are Gigabyte's; other boards use similar ones.

| setting | value | why |
|---|---|---|
| Above 4G Decoding | **Enabled** | the card's 32 GB memory window (BAR) must be placed above 4 GB |
| CSM (Compatibility Support Module) | **Disabled** | CSM forces Above 4G Decoding off |
| Initial Display Output | **the slot or iGPU driving your monitor** | otherwise the firmware may pick the V620, which has no outputs, and halt |

**Before you test anything, save the settings to a profile on a USB stick** (on Gigabyte:
Save & Exit -> Save Profiles). Many boards restore factory defaults after a failed start, and
defaults turn Above 4G Decoding off again, so the next start fails too and it looks as if the
settings "do not stick". With a saved profile, recovery is one Load Profile.

How a missing setting shows up:

- **No picture, VGA debug LED lit:** the firmware picked the V620 as the display. Fix Initial
  Display Output.
- **Linux hangs while loading amdgpu** (hung-task messages after 120 s, `xgpu_nv_mailbox` or
  `amdgpu_virt_init` in the trace): the 32 GB BAR was not assigned. With its memory unmapped the
  driver mistakes the card for a virtual function and waits forever for a hypervisor. Fix Above
  4G Decoding and CSM. To get into the system meanwhile, add `modprobe.blacklist=amdgpu` at the
  GRUB prompt.

## 3. Operating system

Tested: **Ubuntu 26.04 LTS**, kernel 7.0.0-34-generic. Ubuntu 26.04 ships ROCm 7.1 in its own
archive, so no AMD repository or DKMS driver is needed; the in-kernel amdgpu driver is used.

Install as usual. With CSM disabled the installer boots in UEFI mode, which is what you want.

## 4. Kernel parameters

Edit `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX="amdgpu.ras_enable=0 pci=realloc=off amdgpu.gpu_recovery=1 amdgpu.mcbp=0"
```

then `sudo update-grub` and reboot.

- **`amdgpu.ras_enable=0` turns off the card's VRAM ECC. It needs two reboots.** GDDR6 has no ECC
  chips, so the card keeps the check bits in VRAM: ECC costs 2 GB of memory (30704 MiB usable
  instead of 32752) and about 11% of generation speed. The setting is stored on the card: the
  first boot logs `GECC will be disabled in next boot cycle`, the second logs `GECC is disabled`.
  The line `MEM ECC is active` appears either way and does not tell you anything. This one is
  measured: 21.2 -> 24.0 tokens/s without speculative decoding.
- The other three were set during bring-up and never tested one by one. They are harmless on
  this machine; leave them out if you prefer, and add them back if you see PCI resource or GPU
  reset trouble.

### Power cap: 250 W -> 300 W (optional, about +6% prefill)

The V620 is a 300 W board, but its VBIOS PowerPlay table sets the firmware's power limit to
250 W and disables the overdrive power-limit capability, so the driver reports
`power1_cap_min = power1_cap_max = 250 W` and nothing in sysfs or `amdgpu.ppfeaturemask` can
raise it. During prompt processing the card sits exactly at 250 W, at about 2350 MHz against a
2570 MHz maximum. Generation is memory-bound and does not care; prompt processing does.

**Do not upload a modified PowerPlay table through `pp_table`.** The driver resets the SMU to
apply it, the firmware did not come back, and the GPU was unusable until a reboot (which itself
hung and needed the reset switch).

What works is a 3-line driver patch (`power/amdgpu-v620-300w.patch`): for PCI device `1002:73a1`
it raises only the *reported maximum* to 300 W. The default stays 250 W, and setting the cap
afterwards uses the normal runtime path, a single "set limit" message to the firmware with no
reset. The firmware honors it: prompt processing then draws 280-300 W at about 2440 MHz, pp512
470 -> 491 t/s, pp512 at 64k context 246 -> 269 t/s. Sustained prefill at 300 W runs the card
at its thermal limit (see Cooling in section 1); decode is bandwidth-bound and draws 240 to 260 W
at any cap. The driver accepts caps from 250 to 300 W only.

Build the module (no root needed; the kernel's own headers package supplies the prebuilt pieces,
so `flex`/`bison` are not required):

```
apt-get download linux-source-7.0.0 && dpkg-deb -x linux-source-7.0.0_*.deb x
tar xjf x/usr/src/linux-source-7.0.0.tar.bz2 && cd linux-source-7.0.0
patch -p1 < ~/llama-navi21-furnace/deploy/power/amdgpu-v620-300w.patch
H=/usr/src/linux-headers-$(uname -r)
mkdir -p include/config include/generated arch/x86/include/generated scripts/basic scripts/mod tools/objtool
cp -rL $H/include/config/. include/config/; cp -rL $H/include/generated/. include/generated/
cp -rL $H/arch/x86/include/generated/. arch/x86/include/generated/
cp -L $H/.config .config; cp -L $H/Module.symvers .; cp -L $H/scripts/module.lds scripts/
cp -L $H/scripts/basic/fixdep scripts/basic/; cp -L $H/scripts/mod/modpost scripts/mod/
cp -L $H/tools/objtool/objtool tools/objtool/
touch include/config/auto.conf include/generated/autoconf.h
make -j$(nproc) M=drivers/gpu/drm/amd/amdgpu modules KERNELRELEASE=$(uname -r)
strip --strip-debug -o amdgpu.ko drivers/gpu/drm/amd/amdgpu/amdgpu.ko
```

Copy the headers' files with `cp -L`: their entries are relative symlinks into the common headers
package and dangle when copied as links. Do not copy the whole `scripts/` directory over the
tree's own; only the four prebuilt tools and `module.lds`. Check the result with
`modinfo amdgpu.ko | grep vermagic` (must equal `uname -r` plus the stock module's flags) and
`modprobe --dump-modversions` against the stock module (identical CRCs). The build takes a few
minutes; the patched line is visible with `objdump -d` on `sienna_cichlid_get_power_limit`.

Install and enable the cap at boot:

```
sudo mkdir -p /lib/modules/$(uname -r)/updates
sudo cp amdgpu.ko /lib/modules/$(uname -r)/updates/amdgpu.ko
sudo depmod -a && sudo update-initramfs -u && sudo reboot
# after the reboot: modinfo -n amdgpu shows .../updates/amdgpu.ko and power1_cap_max is 300000000
sudo cp ~/llama-navi21-furnace/deploy/power/v620-powercap /usr/local/sbin/
sudo cp ~/llama-navi21-furnace/deploy/power/v620-powercap.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now v620-powercap
```

The module is unsigned; this works because Secure Boot is off (the kernel marks itself tainted).
**The module is tied to the exact kernel version.** After a kernel package update the driver
silently falls back to the stock module, the maximum is 250 W again and the boot service does
nothing (by design). Rebuild the module for the new version with the recipe above;
`modinfo -n amdgpu` pointing at `updates/` is the quick check that the cap is still in effect.
Revert at any time: delete the file under `updates/`, `depmod -a`, `update-initramfs -u`, reboot.

## 5. ROCm and build tools

```
git clone https://github.com/sixvolts/llama-navi21-furnace.git ~/llama-navi21-furnace
~/llama-navi21-furnace/deploy/scripts/install-packages.sh
```

This installs Ubuntu's `rocm` packages, clang 21 (the compiler llama.cpp is built with), git and
Python; adds you to the `render` and `video` groups (GPU access) and `adm` (kernel log, for the
check script); and creates `~/.venv-llama` with cmake 3.31.6, ninja, numpy and pyyaml. The build
was tested with cmake 3.31.6; Ubuntu 26.04's own cmake 4.2 has not been tried.

**Log out and back in** so the new groups apply.

## 6. Check the GPU

```
~/llama-navi21-furnace/deploy/scripts/check-gpu.sh
```

It must end with `RESULT: OK`. Specifically:

- the V620 is on `amdgpu` at **16.0 GT/s x16**, with a **32768 MiB** BAR
- a `root bus resource` window above 4 GB exists
- `Detected VRAM RAM=32752M` and `GECC is disabled` (if you see `30704M`, reboot once more)
- a compute node with **`simd_count=144`**
- `rocminfo` lists **gfx1030**

## 7. Build llama.cpp

```
~/llama-navi21-furnace/deploy/scripts/build-llama.sh
```

This builds the checkout from step 5 (llama-server, llama-cli, llama-bench, llama-quantize and
llama-perplexity for gfx1030; 2.5 minutes on the 8-core test machine). Set `LLAMA_DIR` to build
another checkout, or a path that does not exist yet to clone the fork there first.

The fork is upstream llama.cpp plus changes measured on this card: matrix-multiply and
flash-attention tuning for RDNA2, fused kernels for Qwen3.5-family models, reduced-vocabulary
drafters, and **speculative sampling** (`--spec-draft-temp`): at temperature above 0, drafts are
sampled from the drafter's own distribution and accepted with probability min(1, p/q), which
keeps the output distribution exactly the model's and gains about 15% over drafts that must match
the model's sample exactly.

Check it sees the card: `~/llama-navi21-furnace/build/bin/llama-cli --list-devices` should print
`ROCm0: AMD Radeon Pro V620 (32752 MiB, ...)`.

## 8. Models

Three files, about 19 GB, go in `~/models`:

| file | size | |
|---|---|---|
| `Swift-Qwen3.8-27B-Q4_K_XL-noIQ.gguf` | 16.6 GiB | the model |
| `dflash-Qwen3.8-27B-Q4_0-d2t64k-swiftxl.gguf` | 1.3 GiB | DFlash2 drafter |
| `mtp-Qwen3.8-27B-d2t64k-swiftxl.gguf` | 1.0 GiB | MTP drafter |

Download them:

```
~/llama-navi21-furnace/deploy/scripts/download-models.sh
```

or rebuild them from the upstream sources (about 80 GB of free space; byte-identical results):

```
~/llama-navi21-furnace/deploy/scripts/make-models.sh
```

What they are ([model card](huggingface/README.md) for details and licenses):

- **The model** is Swift's F16 weights quantized with Unsloth's UD-Q4_K_XL per-tensor recipe and
  importance matrix, with the tensors Unsloth stores in IQ formats stored as Q4_K instead, since
  IQ formats are slow on gfx1030 (`recipes/swift-q4_k_xl.txt`). KL divergence against Swift's
  Q8_0: 0.0092, against 0.0134 for Swift's own Q4_K_M.
- **The drafters** propose tokens that the model checks in one batch, so they change speed, not
  output. Both have their 248k-token output head cut to the 65,536 most likely tokens, using rows
  of the model's own output projection, which makes each drafting step several times cheaper.

The model is licensed under the Swift Open License v1.0: free for personal, research and
educational use, and for commercial use up to US$1M annual revenue. See the model card.

### Publishing your own build to Hugging Face

`huggingface/upload.sh` publishes the three files, the model card and the license files to a
repo of yours. It needs the `hf` CLI, which `scripts/install-packages.sh` puts in
`~/.venv-llama`, and a token:

1. Create a token at https://huggingface.co/settings/tokens with **write** access (a
   fine-grained token scoped to the target repo is enough).
2. Log in once; the token is stored in `~/.cache/huggingface/token` and picked up from then on:

   ```
   ~/.venv-llama/bin/hf auth login
   ```

   or, for a one-off or a script, export it instead of saving it: `export HF_TOKEN=hf_...`.
3. Upload (about 19 GB; resumable if interrupted):

   ```
   HF_REPO=<you>/Swift-Qwen3.8-27B-GGUF ~/llama-navi21-furnace/deploy/huggingface/upload.sh
   ```

`download-models.sh` and `make-models.sh` accept `HF_TOKEN` too, but the sources are public and
they do not need one.

## 9. Web tools and prompt

The search tools, the system prompt and the web UI defaults live in
[llama-webui-tools](https://github.com/sixvolts/llama-webui-tools). Clone it next to this one and
follow its [quick start](https://github.com/sixvolts/llama-webui-tools#quick-start): put a Brave
Search API key in `~/.brave_api_key` and run its `scripts/make-mcp-config.sh`.

```
git clone https://github.com/sixvolts/llama-webui-tools.git ~/llama-webui-tools
~/llama-webui-tools/scripts/make-mcp-config.sh
```

`serve-v620.sh` looks for that checkout at `~/llama-webui-tools` (`TOOLS=` to change it).

## 10. Run the server

```
~/llama-navi21-furnace/deploy/serve-v620.sh
```

and open `http://127.0.0.1:8080`. The first request after start is slow (GPU kernels are
prepared on first use); later ones run at full speed.

Choose the drafter with `SPEC`:

| `SPEC=` | research chat, temp 1.0 | math / code / list / essay, temp 1.0 | good for |
|---|---|---|---|
| `dflash` (default) | 42-43 t/s | 76 / 54 / 69 / 40 t/s | mixed use |
| `mtp` | 44 t/s | 62 / 54 / 64 / 43 t/s | mostly chat |
| `none` | 24 t/s | about 24 t/s | |

Numbers are from this machine, warm server, the model card's sampling (temperature 1.0,
top-p 0.95, top-k 20), at short context. With a long context the two drafters part: on a
400-word summary of a long document, MTP holds 48 t/s at 40k to 78k tokens and 44 t/s at 120k,
while DFlash2 gives 43, 40 and 33 t/s at those depths (its drafter runs its own attention over the
context, so each draft step grows with depth). The test machine runs `SPEC=mtp` for that reason:
its sessions are research chats with long tool results. Plain decode is 21, 19 and 17 t/s at
those depths; full tables in `PROGRESS-phoebe.md`, section 25.

To use it from other machines on your network:

```
HOST=0.0.0.0 CORS=http://<this machine's address>:8080 ~/llama-navi21-furnace/deploy/serve-v620.sh
```

With the web tools enabled, anyone who can reach the server can make it search the web and fetch
pages with your API key. Only bind to `0.0.0.0` on a network you trust, or add `--api-key`
(see [Security](https://github.com/sixvolts/llama-webui-tools#security)).

Other settings (`CTX`, `THREADS`, `PORT`, paths) are described at the top of `serve-v620.sh`;
extra arguments are passed to llama-server.

### Sampling settings

`serve-v620.sh` starts the server with the Swift model card's sampling: temperature 1.0,
top-p 0.95, top-k 20, min-p 0, no repetition penalty (1.0) and no presence penalty (0). The
GGUF carries the same values as metadata (`general.sampling.temp`, `.top_p`, `.top_k`), and the
chat template's default reasoning effort is `xhigh`, the level Swift's benchmarks were run at,
so nothing has to be sent for thinking either. To see what the server is using:

```
curl -s localhost:8080/props | python3 -c "import json,sys; p=json.load(sys.stdin)['default_generation_settings']['params']; print({k: p[k] for k in ('temperature','top_p','top_k','min_p','repeat_penalty','presence_penalty')})"
```

The base model's card (Qwen3.5-27B) suggests a presence penalty of 1.5 for general
thinking-mode use; Swift's card and its benchmark runs use 0, which is what the launcher uses.
llama-server's penalties look at the last 64 tokens (`--repeat-last-n`), not the whole reply as
in vLLM, so the two are not the same setting anyway. Change any of these on the command line
(`--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, passed through by the
launcher) or per request. The drafters take the request's sampling as it is: with
`--spec-draft-temp 1.0` a sampled request accepts drafts with the p/q rule, so the output
distribution is the model's own at whatever temperature was asked for; temperature 0 uses
exact-match verification.

**The web UI's own settings.** In the settings dialog ("Sampling & Penalties") the sampling
fields are empty by default, which means "use the server's value", and the dialog shows the
server's value for each. A field you fill in is sent with every request from that browser and
overrides the server; "Reset to default" empties it again. These settings live in the browser
(localStorage), so every browser has its own, and a browser that used an older build of the UI
may still hold that build's hard-coded defaults (temperature 0.8, top-k 40, min-p 0.05) and keep
sending them. After updating the server, open the settings dialog once in each browser and
reset any filled-in sampling field.

## 11. Start at boot

```
mkdir -p ~/.config/systemd/user
cp ~/llama-navi21-furnace/deploy/systemd/llama-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server
sudo loginctl enable-linger $USER     # run without anyone logged in
```

Edit the `Environment=` lines in the copied file to change the drafter, address or port. Logs:
`journalctl --user -u llama-server -f`.

## 12. Troubleshooting

| symptom | cause |
|---|---|
| no picture, VGA LED lit | Initial Display Output points at the V620 (section 2) |
| boot hangs loading amdgpu | 32 GB BAR not assigned: Above 4G Decoding / CSM (section 2) |
| settings keep reverting | failed starts restore BIOS defaults; load your saved profile |
| 30704 MiB VRAM, generation ~11% slow | ECC still on: `amdgpu.ras_enable=0`, then reboot twice (section 4) |
| prompt processing slower after a kernel update | the patched amdgpu module no longer matches; the cap is back at 250 W. Rebuild it (section 4) |
| `rocminfo` fails, no ROCm device | not in the `render`/`video` groups, or not logged out since |
| server fails at start with an out-of-memory error | an environment variable named `TEMP` is set: ROCm reads it as its temporary directory. Unset it. |
| first reply slow, later ones fast | expected: kernels are prepared on first use |
