import argparse 

p = argparse.ArgumentParser()
p.add_argument('--num', type=int)
p = p.parse_args()
print(p)
