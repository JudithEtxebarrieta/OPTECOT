import os
import sys


os.chdir("WindFLO")
sys.path.append(sys.path[0] + '/../WindFLO/Examples/Example1')
print(sys.path)
from example1 import main

main(seed=2, accuracy=1.0)

main(seed=2, accuracy=0.5)

main(seed=2, accuracy=0.25)