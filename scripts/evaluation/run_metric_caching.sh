ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SPLIT=test
# SPLIT=trainval

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path="${ROOT_DIR}/exp/metric_cache" \
scene_filter.frame_interval=1 \
