import math
import sys
import re

import numpy as np
from collections import Counter

class CYK:
    
    def rules(self, folder_path):
        cfg_rules = open(folder_path, 'r')
        lines = [];
        for line in cfg_rules:
            word = line.strip().split('\t')
            if line.strip():
                lines.append(word)

        print(lines)
    def randsentence(self):
        return 1
    
    def CYKParser(self):
        return 1


classCYK = CYK()

classCYK.rules("./Assignment3/cfg.gr")

