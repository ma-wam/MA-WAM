# Release verification — 2026-09-30

A separate fresh Python 3.8 environment installed requirements.txt and passed pip check. A world model completed two updates on 20 real Spread Medium episodes and saved a checkpoint. The planner loaded that checkpoint and a short-trained CoFlow policy, then completed a real 25-step MPE episode for each of policy-only and model selection (two candidates, horizon two, one policy sampling step). The planning evaluation also passed in the fresh environment. Five-seed launch expansion and child-failure reporting were checked.

These are installation and short functional checks, not full-budget retraining or reproduction of paper scores. They do not certify every map, data split, historical checkpoint, optional rendering path, or inherited prototype. Simulator binaries/maps and datasets remain external requirements. Training seeds and evaluation seeds are distinct.
