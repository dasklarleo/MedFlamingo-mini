import sys
sys.path.append('/public/bme/home/liuyx7/project/MedFlamingo-mini/metrics') 
import evaluate
from bleu import Bleu

s1 = 'Zelda Legend is a good game.'
s2 = 'Zelda Legend is not a good game.'
blue_metric = Bleu()
print(blue_metric.compute(predictions = [s1],references = [[s2]]))