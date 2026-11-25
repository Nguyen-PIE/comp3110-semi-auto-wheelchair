# Equal contribution

## Pierre Nguyen (z5482784)

- **First person view for Webots simulator:** The **F** key switches to the rider's eye level (1.15 m
  up), then to each phone walker's view, then back to the overview.
- **Eye level red line laser navigator:** In manual mode a red line runs from the chair along its
  heading and locks on to a wall marker as the rider turns, so **G** sends the chair there.
- **Table, plants, chair obstacles in Webots simulator:** Low obstacles, new every trip, that a seated
  rider sees over but the chair's lidar sees and drives round.
- **Lidar sensor view toggle:** **S** draws a fan on the floor showing what the chair's sensors can see,
  with rings round the people it sees.
- **Visual effect for SiT walking NPC prediction and previously walked path:** **T** draws each nearby
  person's predicted track (orange), walked track (blue) and a ring coloured by kind of walker, and
  scores the prediction as SiT does (ADE and FDE).
- **Random landmark marker generator for user to travel to:** Each take-over hangs 6 numbered markers,
  each its own colour and shape, on the farthest walls in sight, for the rider to pick from.
- **Early eval from testing based on comfort metric:** Comfort measures for every trip (stops, hard
  braking, speed near people) and `eval/compare.py`, which compares two runs on the same trips.

## Teng Jian Jeow (Jayden, z5539338)

- **2D to 3D migration:** Moved the simulator into Webots: the 2D floor plans become 3D worlds, and the
  assistant's code and the 2D policy run there unchanged.
- **Added 3D test cases:** The 3D test set: the same trips for every policy, in worlds kept out of
  training, with three kinds of crowd.
- **Added more 3D maps with special events (e.g. hospital with moving beds):** Situations each building
  is known for, such as an emergency trolley rushing through the hospital or a forklift in the
  warehouses.
- **AWS headless training configuration:** Webots running without a screen on the AWS machine, so
  training and testing can run there.
- **Model testing (wheelchair) in AWS before SiT and after SiT:** Tested the chair's policy on AWS with
  the rule walkers, then again with the walkers trained on SiT.
- **Reduce spinning chair fixes:** The rule that stops the chair turning round on the spot, checked with
  `spin_check.py` (on the spot 3% of the time).
- **Found SiT dataset to train people in simulation:** SiT (Hanyang University, NeurIPS 2023) records
  real people walking round a robot.
- **Train people using SiT dataset:** Retrained the Trajectron++ walk model on SiT, which made its
  predictions 17 to 25% closer to real people than the first model.

## Aung Khant Kyaw (Scott, z5432527)

- **Initial 2D setup with SocNavGym:** The repo, the one-command setup, the SocNavGym scenarios with
  fixes for its bugs, the hospital, office and warehouse floor plans, and the interactive demo with
  take-over control.
- **Rule-based walking people to train the wheelchair:** People who keep a destination and walk an A*
  route to it, planned around the chair, instead of freezing in front of it.
- **Setting up Reinforcement Learning loop to train the wheelchair:** Behaviour cloning from a cautious
  hand-written rule, then PPO. It trained the first model, and the latest model was trained the same
  way, then fine-tuned in 3D.
- **Added Open world scenario to make it realistic with directories and signs (ArUco tags plus OCR) in
  the map:** The chair finds a named room without the floor plan, as a visitor would, and reached it
  in 10 of 10 hospital trips.
- **Added OCR to detect the directories and signs:** RapidOCR reads the words on each sign, and each
  directory row by row, from the chair's camera.
- **Train a YOLO11n sign detector as an alternative to the tags:** It finds 91% of signs in trained
  buildings and 90% in a new one, against 49% and 45% for coloured tags.
- **Voice recognition system with gpt-oss-120b and offline recognition (Vosk) to navigate and run the
  commands:** Exact commands are recognised offline, and anything else is worked out by the model, which
  was right on 92% of 114 test requests with no wrong moves.
- **Proxemics to identify different personal spaces to improve collision distances:** Hall's personal
  space zones round the chair, an idea from mentor Dr Francisco Cruz, measured on every trip, offered as
  a training reward and drawn on the floor with **Z**.
