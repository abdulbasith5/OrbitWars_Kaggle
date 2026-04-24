import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('orbit-wars-reinforcement-learning-tutorial.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
# Print cells 0-20 in full detail
for i, cell in enumerate(nb['cells']):
    if i > 26: break
    src = ''.join(cell['source'])
    if not src.strip(): continue
    ctype = cell['cell_type']
    print("=== CELL", i, "[" + ctype + "] ===")
    print(src)
    print()
