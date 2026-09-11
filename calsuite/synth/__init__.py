"""Synthetic data generators -- one per area (sensor.py here; lens.py,
color.py, display.py belong to their respective waves). Every generator is
seeded (docs/design.md house rule 4: "every analysis has a synthetic round
trip") so a test can generate data from known parameters, run the real
analysis on it, and assert the parameters come back.
"""
