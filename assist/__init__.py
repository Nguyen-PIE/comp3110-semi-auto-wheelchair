"""The shared-control assistant: what runs on the chair.

tracker.py   estimates each nearby person's speed and direction, and predicts collisions
sensors.py   laser-style distances to walls and furniture, from the floor plan or room plan
controller.py  route following, adjusted by the learned policy (speed and steering)
"""
