from main import agent

base_obs = {
    'player': 0,
    'angular_velocity': 0.02,
    'planets': [
        [0, 0, 20.0, 20.0, 5.0, 50, 4],
        [1, 0, 30.0, 25.0, 4.0, 40, 3],
        [2, -1, 45.0, 45.0, 5.0, 3, 2],
        [3, -1, 60.0, 65.0, 4.0, 4, 2],
        [4, -1, 70.0, 30.0, 6.0, 5, 3],
        [5, 1,  80.0, 75.0, 6.0, 12, 4],
        [6, 1,  85.0, 20.0, 5.0, 8, 3],
    ],
    'fleets': [],
    'comets': []
}

for step, label in [(5, "Grab"), (25, "Blitz"), (55, "Mid"), (90, "Late")]:
    obs = dict(base_obs)
    obs['step'] = step
    moves = agent(obs)
    print(label + " (step " + str(step) + "): " + str(len(moves)) + " moves -> " + str(moves))

# Local eval
print("\nRunning local eval...")
import subprocess, sys
result = subprocess.run([sys.executable, "local_eval.py"], capture_output=True, text=True)
print(result.stdout[-800:] if len(result.stdout) > 800 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-400:])
