# Keeping going: the rules and their results

The rules around the policy that deal with blocked ways, and their test results. The short version is
in [webots/README.md](../webots/README.md#keeping-going-ask-walkers-go-round-or-wait).

## The problem

The chair should not wait for a gap that never comes, or drive on into someone about to cut across
it. The 3D test set before these rules (10 November, 540 trips) gave 7 failures, all in busy or
events crowds: 6 timeouts and 1 crash. Eight rules around the policy keep it going
(`assist/controller.py`, `baselines/route_follower.py`).

## The rules

1. **Excuse me, to walkers only.** A walker within 60 cm in front (45 degrees either side) who has
   held the chair up for 3 s is asked to make way ("Excuse me" on the screen), and again every 3 s.
   They walk to the nearest spot clear of the chair's next 5 m that they can reach without brushing
   past anyone, wait there until the chair has passed (8 s at most) and walk on. If someone steps
   into their way, they walk on after 1.5 s. People standing to talk or wait are never asked to
   break up: the route goes round them.
2. **Round someone who keeps it waiting.** Held up 4 s by someone right in front (even someone
   shuffling about, not looking), it plans round them for 10 s. Not round someone it has just asked
   to step aside.
3. **Keep to one way.** It keeps its route while nobody has stood on it for 5 s and no new route is
   3 m shorter. Before, in two-way traffic past a row of benches, it turned back and forth between
   the two sides whenever someone paused.
4. **Let a faster walker pass.** Someone overtaking right beside it (within 30 cm) is let ahead: the
   chair slows to 15% until they are past.
5. **Wait for someone cutting across.** The safety stop looks at where people are now. Driving on,
   the chair also follows walkers over the next 0.5 s and waits if one would come within 10 cm, and
   it waits for anyone walking in towards its front half within 25 cm.
6. **Half speed near walkers.** With someone walking within 50 cm in front or beside, half speed at
   most: the tracker can be 15 to 23 cm behind a walker who turns.
7. **Out of a crush or a wall corner.** Held still 2 s by the safety stop or the wall brake, it
   drives out to open space: a way ahead first (it backs off only when every way on is closed), never
   where the wall brake would hold it, and never closer than 10 cm to someone walking.
8. **Do not turn round on the spot.** A new route that goes 120 degrees or more
   from the way the chair is going would send it back the way it came, so it is taken only once the
   chair has waited 8 s for the way on, looking again every second. After turning back it does not
   swing round again for 10 s while its new way is open. Getting out of a crush keeps closer to the
   route, and plans again from where it got out instead of going back for a point on the route behind
   it. `webots/spin_check.py` counts the turning on the spot that is left and what caused it.

## Results

| 540 test trips, the learned 2D policy | before the rules (10 Nov) | 7 rules (10 Nov) | 8 rules, wall brake, rider (18 Nov) |
| --- | --- | --- | --- |
| Reached the goal | 533 (98.7%) | 537 (99.4%) | 537 (99.4%) |
| Timeouts | 6 | 3 | 3 |
| Crashes, chair moving / standing | 1 / 0 | 0 / 0 | 0 / 0 |

The 18 November run (all 8 rules, the wall brake and the seated rider) failed on 3 trips, all events
trips that ran out of time: corridor seeds 10017 and 10018 and hospital seed 10006. Corridor seed
10010, which timed out with 7 rules, reached the goal, and 10017 did not. On average a
trip changed very little: 21.1 s to the goal (21.2 s before), 15.1 m driven (the same), 10.6 hard
brakes (10.5) and 5.5 stops a minute (5.4).

On the 470 trips that reached the goal both times (8 worlds, the depot's results from before were
lost): 20.7 s to the goal (20.5 before), 14.8 m driven (15.3), 1.7 near misses (1.6) per trip, but
more stopping: 10.0 hard brakes (8.8), 2.9 safety stops (2.0) and 5.4 stops a minute (4.7), from
rule 5. The 3 timeouts are corridor and hospital events trips.

### Variation between runs

Across five full reruns while building these rules, 535 to 538 trips reached the goal each time: every change shifts every trip, and a
different 2 to 4 busy or events trips end in a jam (head-on in a one-lane bit of corridor, or both
sides of a row of benches blocked by people waiting where they arrived). A trip's result can also
depend on the trips run before it in the same Webots.

## Changes to the test

The test itself changed a little: walkers can be asked to make way, people standing still are only
placed where they leave the chair 30 cm to pass and never close a way it had, a walker pressed
against the chair takes another way round it at once (before, after 3 s), and people stop 2 cm short
of the chair instead of right against it.

## Limitations

The rules are plain code around the policy. Training the policy to do all of this itself (its actions,
inputs, reward, scenarios and pass criteria) was not done.
