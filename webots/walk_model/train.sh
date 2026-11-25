#!/usr/bin/env bash
# Train the walking model (Trajectron++) on real recordings of people, then save it where the
# simulator finds it (models/walk_trajectron/). Needs no GPU; on a laptop each pass over the data
# takes a few minutes, so 100 passes is an overnight job (or use the AWS machine).
#     bash webots/walk_model/train.sh [passes, default 100] [data, default zara1]
# data:
#   eth, hotel, univ, zara1 or zara2   ETH/UCY, leaving that place out of training, so that
#                                      offline_eval.py can test the model on people it never saw
#   sit                                SiT (people walking around a robot), with its own split of
#                                      scenes; run sit_to_trajectron.py <SiT folder> first
#   sitmix                             SiT plus ETH/UCY (sit_to_trajectron.py <SiT folder> --with-eth)
set -euo pipefail
cd "$(dirname "$0")/../.."
EPOCHS="${1:-100}"
HELD_OUT="${2:-zara1}"
T="$PWD/third_party/Trajectron-plus-plus"
[[ -d "$T/trajectron" ]] || bash webots/walk_model/get_trajectron.sh

if [[ "$HELD_OUT" == sit* && ! -d "$T/experiments/pedestrians/raw/$HELD_OUT/train" ]]; then
  echo "No SiT data converted yet. Run first:  python webots/walk_model/sit_to_trajectron.py <SiT folder>" \
       "$([[ $HELD_OUT == sitmix ]] && echo --with-eth)"
  exit 1
fi
CONF="$T/experiments/pedestrians/models/${HELD_OUT}_attention_radius_3/config.json"
[[ -f "$CONF" ]] || CONF="$T/experiments/pedestrians/models/zara1_attention_radius_3/config.json"   # same settings

if [[ ! -f "$T/experiments/processed/${HELD_OUT}_train.pkl" ]]; then
  echo "==> Preparing the recordings ($HELD_OUT)"
  (cd "$T/experiments/pedestrians" && python process_data.py "$HELD_OUT")
fi

echo "==> Training for $EPOCHS passes (progress below; Ctrl+C stops, the last saved pass is kept)"
tag="_${HELD_OUT}_walk"
(cd "$T/trajectron" && python train.py --train_data_dict "${HELD_OUT}_train.pkl" --eval_data_dict "${HELD_OUT}_val.pkl" \
  --offline_scene_graph yes --preprocess_workers 0 --log_dir ../experiments/pedestrians/models --log_tag "$tag" \
  --train_epochs "$EPOCHS" --augment --conf "$CONF" \
  --device cpu --eval_every 1000 --vis_every 1000 --save_every 5 --seed 123)

run="$(ls -td "$T"/experiments/pedestrians/models/*"$tag" | head -1)"
python webots/walk_model/export.py "$run"
python webots/walk_model/offline_eval.py --scene "$HELD_OUT"
