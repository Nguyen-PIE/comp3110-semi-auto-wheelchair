# Proxemics: personal space around the chair

Proxemics is the study of the distances people keep from each other. This page describes how the
project draws personal space zones around the chair, measures how long people spend in each zone on
every trip, and offers the zones as an optional training reward. The reward alone made no clear
difference, but the latest model, fine-tuned with it and with walkers trained on real people, had no
collisions in 540 trips, and the zone measures are part of every evaluation.

## Background

The idea came from the team's mentor, Dr Francisco Cruz (UNSW), who suggested looking at proxemics: the
study of the distances people keep from each other, which Hall split into intimate, personal, social and
public zones [1]. His own research with Millán-Arias and Fernandes studies proxemics in reinforcement
learning, where an agent navigating towards a person learns how close it can comfortably come from the
feedback a simulated person gives on each approach [2, 3]. Their agent learns where the uncomfortable
region is. Here the zone sizes are fixed instead, from Hall and from studies of robots passing people
[4, 5], and the reward charges the chair for coming into them.

The original reward penalised a person closer than 0.5 m to a 0.4 m circle. That circle is too short at
the front of the real chair and too wide at its sides. Studies of robots passing people found that
people want more space in front of them while they move and that their comfort drops as the robot gets
faster [4], and that passing behind someone is worse than passing in front [5]. Work on robotic
wheelchairs and personal mobility vehicles adds that a comfortable path has to suit the rider as well as
the people passed [6, 7]. In the held-back 3D test the first model spent 4.7 s a trip within
0.5 m of someone, against 4.1 s for the cautious rule.

## Design

### The zones

`assist/proxemics.py` puts Hall's four zones [1] around the chair as ellipses, grown out from the
chair's real outline (1.1 m long, 0.65 m wide). The front of each ellipse stretches with speed, up to
twice as deep at full speed, because people want more room in front of something coming at them
faster [4]. People stay circles of their shoulder width, and the collision model is not touched. A
person is in the zone their nearest edge reaches.

| Zone (gap at rest) | Reward per 0.1 s step (`--reward proxemic`) | Evaluation column |
| --- | --- | --- |
| Intimate, under 0.45 m | up to -0.25 x speed, as the original reward | `intimate_s`, `intimate_entries`, `intimate_moving_in_s`, `speed_in_intimate` |
| Personal, 0.45 to 1.2 m | up to -0.02 x speed | `personal_s` |
| Social, 1.2 to 3.6 m | 0 | `social_s` |
| Public, 3.6 to 12 m | 0 | `public_s` |
| Nobody within 12 m | 0 | `nobody_s` |

### The reward

- The original reward stays the default. The proxemic reward is only used with `--reward proxemic`
  (`G19_REWARD=proxemic`).
- People behind the chair's back edge are left out of the reward, because the chair cannot reverse and
  driving on takes it away from them.
- A stopped chair costs nothing, so people walking past a waiting chair are free.
- The personal zone penalty never outweighs the progress reward (+0.12 a step at full speed), so it
  shapes the path without stopping the chair.
- The social zone has no penalty, because in a 2 m corridor anyone passing is inside it, and a cost for
  everyone nearby risks the "freezing robot" problem, where every way forward looks too costly and the
  robot stops [8].

### The measures

- `eval/comfort.py` adds the columns in the table to every evaluation. The five times add up to the trip
  time, and each step goes to the zone of the nearest person, including people behind the chair.
- `eval/metrics.py` counts `personal_space_compliance` in the evaluation itself, so it is measured in
  Webots too.

## Seeing the zones

`python webots/run.py fp_hospital --zones` draws the zones on the floor round the chair, and **Z** turns
them on and off (`webots/zoneview.py`). Red is intimate, orange personal and yellow social, and the
front stretches as the chair speeds up. They are drawn on the 3D view only, so nothing else in the
simulator changes. The zones and the sensor fan are never shown together, because the two see-through
layers flicker where they overlap.

## Training with the proxemic reward

### Method

The first model, `models/assist_ppo.zip`, was trained in 2D and is driven in Webots, so the
proxemic reward followed the same path: training in 2D, fine-tuning in Webots with the walkers trained on
SiT, and testing in 3D on the 540 validation trips. The original reward was trained the same way beside
it, so the two differ only in the reward. It ran on the AWS machine.

```bash
python training/train_assist.py --reward old      --seed 0 --out models/r_old_2d.zip
python training/train_assist.py --reward proxemic --seed 0 --out models/r_prox_2d.zip
python webots/train.py --start-from models/r_old_2d.zip  --reward old      --walkers learned --envs 4 --timesteps 100000 --out models/sit_old.zip
python webots/train.py --start-from models/r_prox_2d.zip --reward proxemic --walkers learned --envs 4 --timesteps 100000 --out models/sit_prox.zip

# the 540 validation trips, with SiT walkers and with rule walkers
python webots/evaluate.py --driver auto --walkers learned --parallel 2
python webots/evaluate.py --driver rule --walkers learned --parallel 2
for m in r_old_2d r_prox_2d sit_old sit_prox; do
  python webots/evaluate.py --model models/$m.zip --walkers learned --parallel 2
  python webots/evaluate.py --model models/$m.zip --walkers rules --parallel 2
done
```

### Pass criteria

Each proxemic model is compared with the model trained the same way with the original reward
(`python eval/compare.py --table data/episodes/*.parquet`). It passes if:

- success is at least 99%
- person collisions are no higher than with the original reward
- `intimate_s` is down at least 25%
- hard brakes, delay ratio and timeouts are up no more than 10%

### Results

The proxemic reward did not pass. Against the model trained with the original reward, with the SiT
walkers, the time with someone in the intimate zone fell by 3% after the fine-tune and rose by 3% in 2D,
neither a clear change. The one clear change was 6.9% less time driving into someone's intimate zone
after the fine-tune, while the chair weaved about twice as often. Someone is in the intimate zone for
about 40% of every trip whatever drives the chair, the cautious rule included, because people come close
in corridors, so a 25% cut was not a realistic bar for a reward change alone. The fine-tuned proxemic policy is kept as
the latest model (`models/sit_prox.zip`): with the SiT walkers it reached the goal in 539 of 540 trips with
no collisions, against 538 for the first model. The full results are in
[evals.md](evals.md#retraining-with-the-sit-walkers-and-the-proxemic-reward).

## Limitations

- The zone sizes are fixed from the studies above. No rider or passer-by has rated them.
- Each reward was trained with one seed, so a small effect could not be told apart from run-to-run
  variation.
- Learning the zone sizes from feedback, as in Dr Cruz's work [2, 3], would be a natural next step, for
  example from the rider's take-overs.

## References

1. Hall, E. T. (1966). *The Hidden Dimension*. Doubleday.
2. Millán-Arias, C., Fernandes, B., & Cruz, F. (2021). Learning proxemic behavior using reinforcement
   learning with cognitive agents. Extended abstract, Human-Aligned Reinforcement Learning for
   Autonomous Agents and Robots workshop, IEEE ICDL 2021. https://arxiv.org/abs/2108.03730
3. Millán-Arias, C., Fernandes, B., & Cruz, F. (2023). Proxemic behavior in navigation tasks using
   reinforcement learning. *Neural Computing and Applications*, 35(23), 16723-16738.
   https://doi.org/10.1007/s00521-022-07628-0
4. Neggers, M. M. E., Cuijpers, R. H., Ruijten, P. A. M., & IJsselsteijn, W. A. (2022). The effect of
   robot speed on comfortable passing distances. *Frontiers in Robotics and AI*, 9, 915972.
   https://doi.org/10.3389/frobt.2022.915972
5. Neggers, M. M. E., Cuijpers, R. H., Ruijten, P. A. M., & IJsselsteijn, W. A. (2022). Determining
   shape and size of personal space of a human when passed by a robot. *International Journal of Social
   Robotics*, 14(2), 561-572. https://doi.org/10.1007/s12369-021-00805-6
6. Morales, Y., Miyashita, T., & Hagita, N. (2017). Social robotic wheelchair centered on passenger and
   pedestrian comfort. *Robotics and Autonomous Systems*, 87, 355-362.
   https://doi.org/10.1016/j.robot.2016.09.010
7. Isono, Y., Yoshitake, H., & Shino, M. (2022). Passenger's comfort recognition during autonomous
   navigation of personal mobility vehicles in crowded pedestrian spaces. In *Proceedings of the 17th
   International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and
   Applications, Volume 2: HUCAPP* (pp. 58-67). SCITEPRESS. https://doi.org/10.5220/0010849700003124
8. Trautman, P., & Krause, A. (2010). Unfreezing the robot: Navigation in dense, interacting crowds. In
   *2010 IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 797-803). IEEE.
