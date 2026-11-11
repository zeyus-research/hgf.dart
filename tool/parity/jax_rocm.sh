#!/bin/sh
# JAX on an AMD consumer GPU (RDNA3, gfx110X) for the GPU comparisons,
# without a system ROCm install: AMD's TheRock wheels from repo.amd.com in
# their own venv (tool/parity/.venv-rocm), plus pyhgf from ../pyhgf.
#
#     tool/parity/jax_rocm.sh setup            # once (~4 GB of wheels)
#     tool/parity/jax_rocm.sh python args...   # python with JAX on the GPU
#     dart run tool/bench_compare.dart --gpu --jax-gpu tool/parity/.venv-rocm/python-rocm
#
# Pitfalls this script handles:
# - PyPI's jax-rocm7-plugin `with-rocm` extra pulls the Instinct (MI-series)
#   builds; gfx110X builds are only on AMD's index, and lag JAX (0.9.x there,
#   while tool/parity's CPU JAX is newer).
# - The plugin links librocm_sysdeps_hwloc.so.5 and _pciaccess.so.0, which
#   no gfx110X wheel ships. pciaccess is linked but unused (an empty stub
#   does); hwloc is used for NUMA host memory, so a shim exports the 14
#   functions under AMD's symbol version and forwards them to the system's
#   hwloc 2.x (libhwloc.so.15, same ABI).
# - gfx1101/gfx1102 (RX 7600/7700/7800) need HSA_OVERRIDE_GFX_VERSION=11.0.0.
# - XLA's Triton GEMM autotuner fails on some f32 matmuls there ("Autotuner
#   could not compile any configs"), so the wrapper turns Triton GEMMs off
#   (hipBLASLt/rocBLAS do them instead).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
V=$HERE/.venv-rocm
INDEX=${ROCM_INDEX:-https://repo.amd.com/rocm/whl/gfx110X-all/}
ROCM=${ROCM_VERSION:-7.13.0}
JAX=${JAX_ROCM_VERSION:-0.9.0}

setup() {
  uv venv -p 3.13 "$V"
  VIRTUAL_ENV=$V uv pip install --index-strategy unsafe-best-match \
    --extra-index-url "$INDEX" \
    "jax==$JAX" "jaxlib==$JAX+rocm$ROCM" \
    "jax-rocm7-plugin==$JAX+rocm$ROCM" "jax-rocm7-pjrt==$JAX+rocm$ROCM" \
    "rocm[libraries]==$ROCM" numpyro "$HERE/../../../pyhgf"

  mkdir -p "$V/stubs" "$V/shim"
  echo "" | cc -shared -o "$V/stubs/librocm_sysdeps_pciaccess.so.0" \
    -Wl,-soname,librocm_sysdeps_pciaccess.so.0 -x c -
  names="hwloc_alloc_membind hwloc_bitmap_alloc hwloc_bitmap_free
    hwloc_bitmap_isincluded hwloc_free hwloc_get_area_memlocation
    hwloc_get_cpubind hwloc_get_nbobjs_by_depth hwloc_get_obj_by_depth
    hwloc_get_type_depth hwloc_set_cpubind hwloc_topology_init
    hwloc_topology_load hwloc_topology_set_flags"
  n=$(echo $names | wc -w)
  {
    echo '#include <dlfcn.h>'
    echo '#include <stdio.h>'
    echo "void *hwloc_shim_ptrs[$n];"
    printf 'static const char *names[] = {'
    for s in $names; do printf '"%s",' "$s"; done
    echo '};'
    echo '__attribute__((constructor)) static void init(void) {'
    echo '  void *h = dlopen("libhwloc.so.15", RTLD_NOW | RTLD_LOCAL);'
    echo '  if (!h) { fprintf(stderr, "hwloc shim: %s\n", dlerror()); return; }'
    echo "  for (int i = 0; i < $n; i++) hwloc_shim_ptrs[i] = dlsym(h, names[i]);"
    echo '}'
  } >"$V/shim/shim.c"
  {
    echo '.text'
    i=0
    for s in $names; do
      echo ".globl $s"
      echo ".type $s,@function"
      echo "$s:"
      echo "  movq hwloc_shim_ptrs@GOTPCREL(%rip), %rax"
      echo "  jmp *$((8 * i))(%rax)"
      i=$((i + 1))
    done
    echo '.section .note.GNU-stack,"",@progbits'
  } >"$V/shim/tramp.S"
  echo 'AMDROCM_SYSDEPS_1.0 { global: hwloc_*; local: *; };' >"$V/shim/ver.map"
  cc -O2 -fPIC -shared -o "$V/stubs/librocm_sysdeps_hwloc.so.5" \
    -Wl,-soname,librocm_sysdeps_hwloc.so.5 \
    -Wl,--version-script="$V/shim/ver.map" \
    "$V/shim/shim.c" "$V/shim/tramp.S" -ldl

  sp=$("$V/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  core=$sp/_rocm_sdk_core
  libs=$(ls -d "$sp"/_rocm_sdk_libraries_*)
  cat >"$V/python-rocm" <<EOF
#!/bin/sh
export LD_LIBRARY_PATH=$V/stubs:$core/lib:$core/lib/rocm_sysdeps/lib:$libs/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export HSA_OVERRIDE_GFX_VERSION=\${HSA_OVERRIDE_GFX_VERSION:-11.0.0}
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false \${XLA_FLAGS-}"
export ROCM_PATH=$core
export JAX_PLATFORMS=\${JAX_PLATFORMS-}
exec $V/bin/python "\$@"
EOF
  chmod +x "$V/python-rocm"
  "$V/python-rocm" -c 'import jax; d = jax.devices(); print(d); assert d[0].platform != "cpu"'
}

case "${1:-}" in
setup) setup ;;
python)
  shift
  exec "$V/python-rocm" "$@"
  ;;
*)
  echo "usage: $0 setup | python args..." >&2
  exit 64
  ;;
esac
