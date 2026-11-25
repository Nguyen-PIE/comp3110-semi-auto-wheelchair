# Training and testing on the AWS machines

The group shares **one** machine: a c6a.4xlarge-sized instance (16 vCPUs, which is 8 physical cores,
about 30 GiB, no GPU, Ubuntu 24.04, about 76 GB of disk), logged in as `g19-admin` through the CSE login
host. Each of us works in our own folder on it (`~/pierre`, `~/jayden`, `~/scott`, and so on), with our own
clone of the repo inside. This page is the runbook: set up, measure, run overnight, share results.
Deadline: **23 November**.

## 0. Sharing the one machine

1. It has 8 physical cores for everyone together, so a long Webots run slows everyone else's. Before
   starting anything heavy, look at who is on it: `who`, `tmux ls`, `top` (press `q` to leave it).
2. **Pick your own Webots port range.** `webots/train.py` and `webots/evaluate.py` start at port 1234 and
   count up (one port per Webots copy), and every Webots uses `/tmp/webots/g19-admin/<port>`. Two people on
   the default ports connect to each other's Webots. Use `--port`, with a gap of at least `--envs` between
   people, for example `--port 1300` for Pierre, `--port 1500` for Jayden and `--port 1700` for Scott.
   `webots/cloud/benchmark.py` uses 1400 by default, which is why it needs the whole machine to itself.
3. Do not `pkill webots` without a port: it kills everyone's runs.
4. Stay inside your own folder, so models and `data/episodes` do not overwrite each other's.
5. Do not stop the instance while anyone has something running (check with `tmux ls`). A stopped
   machine keeps its disk, but its public address changes when it starts again.
6. Reach GitLab from the machine with agent forwarding (`ssh -A`, after `ssh-add` on your own computer).
   Never copy a private key onto it: it is a shared account, so the key would be everyone's.

## 1. Set up the machine

```bash
git clone git@gitlab.cse.unsw.edu.au:coursework/comp3110/26t3/students/groups/G19.git G19 && cd G19
git checkout <the branch the group agrees on>
bash webots/cloud/setup_ubuntu.sh
```

`setup_ubuntu.sh` has never been run on AWS. If it fails, paste the last 20 lines in the group chat, fix
it in a merge request, and everyone else uses the fixed version. It ends by driving two trips headless;
if that works, the machine is ready.

## 2. Measure before running anything long

The right number of Webots copies is not 14. The machine has 8 physical cores, and Webots draws its lidar
on the CPU.

```bash
tmux new -s bench
conda activate wheelchair            # or: source .venv/bin/activate
python webots/cloud/benchmark.py                           # 4, 8, 12, 14 copies, 4 minutes each (about 20 min)
python webots/cloud/benchmark.py --env LP_NUM_THREADS=1    # does limiting Mesa's threads help? (unchecked idea)
python webots/cloud/benchmark.py --walls lidar map --envs 8   # what does the lidar cost?
```

Paste the printed table in the group chat. Use the smallest number of copies within 10% of the best
(the script prints it) as `--envs` for everything below. Also glance at `free -h` during a run; if memory
gets close to 32 GiB, use fewer copies.

## 3. Run it

Always inside tmux, so it keeps going when you disconnect (detach with Ctrl-b then d, return with
`tmux attach -t train`).

```bash
# Baseline (first job for everyone, 100 trips per world): the numbers every later change is judged against
python eval/run_episodes.py --scenario all --episodes 100 --driver assist --save-traces
python eval/run_episodes.py --scenario all --episodes 100 --driver rule --save-traces
python webots/evaluate.py --parallel 4 --episodes 100 --model models/assist_ppo_webots.zip --save-traces

# Fine-tune in Webots (only for a model chosen from the 2D sweeps)
python webots/train.py --envs <N from the benchmark> --timesteps 100000 --out models/<run name>.zip
```

2D training for the reward sweeps is `python training/train_assist.py` (about 45 minutes on 2 cores, so
about 7 runs at once on the whole machine, which is only 2 or 3 each when three of us share it). It does not take reward weights on the command line yet; adding that
is the first job of whoever owns the reward.

## 4. Sharing results

- Name every run for what it changes, and put the settings in the name, for example
  `jerk0.3_decel0.8_seed2`, so a file explains itself a week later.
- Results are small (a few hundred KB). After a run, copy `data/episodes/*.parquet`, `data/traces/*.parquet`,
  `data/benchmarks/*.csv` and the model `.zip` to one shared folder, in a subfolder with your name:
  `scp -r ubuntu@<ip>:G19/data/episodes ./results/<yourname>/`.
- Compare many runs in one table: `python eval/compare.py --table results/*/*.parquet`. Compare two on the
  same trips with `python eval/compare.py A.parquet B.parquet --by scenario`.
- Commit code changes, never results or other people's models, on one shared branch with merge requests;
  an instance carrying its own edits is how two people end up reporting different numbers for "the same" run.

## 5. Plan to the deadline

| Dates | What | Owner |
| --- | --- | --- |
| 18 to 19 Nov | Anything from the first plan that is not finished: the machine set up and benchmarked, baselines with the comfort columns (section 3), the 2D sweeps. Then choose the best settings from the Pareto front (low stops and braking, small delay, no new collisions) on seeds 10000 to 10199 | all |
| 19 to 20 Nov | Fine-tune the winner in Webots on one machine; the others run 100-trip evals and the scenario suite; small human preference test | split |
| 21 to 22 Nov | Freeze models. Final evaluation on seeds 20000 and up, once. Write up. | all |
| 23 Nov | Deadline | |

The first version of this plan ran from 6 November (set up and benchmark, baselines, the scenario suite and
the reward settings fixed on 11 November, overnight 2D sweeps 11 to 15 November). Anything in that list
that is not done is now squeezed into 18 to 19 November, so cut by the rule below rather than let it push
the freeze past 22 November.

Fallback: if fine-tuning in Webots does not help (the 50,000-step run did not), submit the best 2D-trained
policy evaluated in Webots. That pipeline already works.

Cut first if time runs out: a new trajectory predictor or a changed observation. Either one means
redoing behaviour cloning and RL from scratch.
