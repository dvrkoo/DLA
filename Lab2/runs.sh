#!/bin/bash

EPISODES=1000

for algo in reinforce ppo; do
  for env in CartPole-v1 LunarLander-v2; do
    for mode in worst mid best; do
      case $mode in
        worst)
          TEMP=2.0
          GAMMA=0.5
          ;;
        mid)
          TEMP=1.0
          GAMMA=0.9
          ;;
        best)
          TEMP=0.8
          GAMMA=0.99
          ;;
      esac

      for det in false true; do
        DET_FLAG=""
        DET_SUFFIX=""
        if $det; then
          DET_FLAG="--deterministic"
          DET_SUFFIX="_det"
        fi

        echo "Running $algo on $env ($mode-case, deterministic=$det)"
        python main.py \
          --algo $algo \
          --env $env \
          --episodes $EPISODES \
          --temperature $TEMP \
          --gamma $GAMMA \
          $DET_FLAG \
          --gif gifs/${algo}_${env}_${mode}${DET_SUFFIX}.gif
      done
    done
  done
done

