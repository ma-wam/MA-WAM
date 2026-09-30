"""Convert MPE per-agent per-seed data to flat format for WorldModelTrainer."""
import os
import numpy as np
import argparse

def convert(env_name, src_root, dst_root):
    src_base = os.path.join(src_root, env_name)
    splits = [d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))]

    for split in splits:
        src_split = os.path.join(src_base, split)
        dst_split = os.path.join(dst_root, env_name, split)

        if os.path.exists(os.path.join(dst_split, "obs.npy")):
            print(f"[SKIP] {env_name}/{split} already converted")
            continue

        seed_dirs = sorted([d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))])
        if not seed_dirs:
            print(f"[SKIP] {env_name}/{split} no seed dirs")
            continue

        # Detect n_agents from first seed
        first_seed = os.path.join(src_split, seed_dirs[0])
        n_agents = len([f for f in os.listdir(first_seed) if f.startswith("obs_")])

        all_obs, all_acts, all_rews, all_pl = [], [], [], []

        for sd in seed_dirs:
            sd_path = os.path.join(src_split, sd)
            obs_list = [np.load(os.path.join(sd_path, f"obs_{i}.npy")) for i in range(n_agents)]
            act_list = [np.load(os.path.join(sd_path, f"acs_{i}.npy")) for i in range(n_agents)]
            rew_list = [np.load(os.path.join(sd_path, f"rews_{i}.npy")) for i in range(n_agents)]
            dones = np.load(os.path.join(sd_path, "dones_0.npy"))

            ep_len = obs_list[0].shape[0]
            # Pad obs to max dim across agents, then stack
            max_obs_dim = max(o.shape[-1] for o in obs_list)
            obs_padded = []
            for o in obs_list:
                if o.shape[-1] < max_obs_dim:
                    pad = np.zeros((o.shape[0], max_obs_dim - o.shape[-1]), dtype=o.dtype)
                    o = np.concatenate([o, pad], axis=-1)
                obs_padded.append(o)
            obs = np.stack(obs_padded, axis=1)
            acts = np.stack(act_list, axis=1)
            # rews may be 1D (T,) or 2D (T,1)
            rews_clean = [r.reshape(r.shape[0]) if r.ndim > 1 else r for r in rew_list]
            rews = np.stack(rews_clean, axis=1)  # (T, n_agents)

            # Split into episodes by dones
            done_indices = np.where(dones.flatten())[0]
            if len(done_indices) == 0:
                # Fixed episode length - try 25 steps
                ep_length = 25
                n_eps = ep_len // ep_length
                for i in range(n_eps):
                    s, e = i * ep_length, (i + 1) * ep_length
                    all_obs.append(obs[s:e])
                    all_acts.append(acts[s:e])
                    all_rews.append(rews[s:e])
                    all_pl.append(ep_length)
            else:
                start = 0
                for di in done_indices:
                    end = di + 1
                    all_obs.append(obs[start:end])
                    all_acts.append(acts[start:end])
                    all_rews.append(rews[start:end])
                    all_pl.append(end - start)
                    start = end

        # Concatenate
        obs_cat = np.concatenate(all_obs, axis=0)
        acts_cat = np.concatenate(all_acts, axis=0)
        rews_cat = np.concatenate(all_rews, axis=0)
        pl_arr = np.array(all_pl)

        os.makedirs(dst_split, exist_ok=True)
        np.save(os.path.join(dst_split, "obs.npy"), obs_cat)
        np.save(os.path.join(dst_split, "actions.npy"), acts_cat)
        np.save(os.path.join(dst_split, "rewards.npy"), rews_cat)
        np.save(os.path.join(dst_split, "path_lengths.npy"), pl_arr)

        print(f"[DONE] {env_name}/{split}: obs={obs_cat.shape}, acts={acts_cat.shape}, "
              f"rews={rews_cat.shape}, n_eps={len(pl_arr)}, pl_range=[{pl_arr.min()},{pl_arr.max()}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["simple_tag", "simple_world"])
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--src-root",
        default=os.path.join(project_root, "diffuser", "datasets", "data", "mpe"),
        help="directory containing per-agent MPE seed data",
    )
    parser.add_argument(
        "--dst-root",
        default=os.path.join(project_root, "data", "mpe"),
        help="directory for MA-WAM's flattened MPE datasets",
    )
    args = parser.parse_args()

    for env in args.envs:
        convert(env, args.src_root, args.dst_root)
