#!/bin/bash -l
# One time setup on a Myriad login node. Run from ~/Scratch/lambda-audit/.
#
# The shebang carries -l on purpose. `module` is a shell FUNCTION defined when
# a login shell sources the system profile, so a plain `bash setup_myriad.sh`
# fails immediately with "module: command not found". The block below makes it
# work under either invocation, so this cannot bite again.
set -e
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/lmod.sh; do
    if [ -r "$f" ]; then . "$f"; break; fi
  done
fi
if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: 'module' still undefined. Run this as: bash -l setup_myriad.sh"
  exit 1
fi
module purge
module load default-modules 2>/dev/null || true
# PINNED, no fallback. `python3/recommended` is 3.9 and the venv below is built
# against 3.11: loading the fallback would produce a venv whose interpreter
# cannot start ("libpython3.11.so.1.0: cannot open shared object file"). A hard
# failure here is worth far more than a wrong interpreter that runs 240 tasks.
# Confirmed available on Myriad 2026-08-08: 3.4 3.5 3.6 3.7 3.8 3.9 3.11.
module load python3/3.11

python3 -m venv ~/Scratch/lambda-audit/venv
source ~/Scratch/lambda-audit/venv/bin/activate

# Prove which interpreter we are actually in before installing anything into
# it. A loaded pytorch module can make a bare `python3 -c "import torch"` print
# a healthy version from /shared/ucl/apps while the venv is broken or absent.
echo "--- interpreter"
python -c "import sys; print(sys.executable); print(sys.version)"
case "$(python -c 'import sys; print(sys.executable)')" in
  "$HOME"/Scratch/lambda-audit/venv/*) echo "  ok, inside the venv" ;;
  *) echo "  ERROR: not the venv interpreter. Aborting."; exit 1 ;;
esac

pip install --upgrade pip
# CPU torch is sufficient. The networks are small and there is no GPU leg in
# the pilot. If the pytorch CPU index is blocked from the login node, the
# fallback line pulls the default wheel instead.
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
pip install popgym numpy

# Freeze the resolved versions. The pilot is preregistered, so the environment
# that produced the numbers has to be recoverable later. Nothing above is
# version-pinned on purpose (pinning untested wheels risks the install), so
# this file is the record of what was actually used.
pip freeze > env_frozen.txt
echo "--- froze $(wc -l < env_frozen.txt) packages to env_frozen.txt"

echo "--- versions"
python -c "import torch, popgym, gymnasium; print('torch', torch.__version__); print('popgym', popgym.__version__ if hasattr(popgym,'__version__') else 'ok'); print('gymnasium', gymnasium.__version__)"

# smoke test, about one minute, must reach a clearly positive return
echo "--- smoke: RepeatPreviousEasy + stack8 at lambda 1.0, 600k steps"
echo "--- expect deployed_return near 0.99, random policy is about -0.54"
python ppo_lambda.py --env RepeatPreviousEasy --arch stack8 --gae-lambda 1.0 \
    --seed 1 --total-timesteps 600000 --eval-every 50 --eval-episodes 10 \
    --final-eval-episodes 30 --out-dir smoke --run-name setup_smoke
echo "SETUP OK if the deployed_return above is greater than 0.9"
