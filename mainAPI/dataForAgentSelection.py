'''
The data needs to be organized in this format to feed it to the AgentSelectionMechanism.py
Return:
#agents_record = {"ETH_ADDRESS":[GAMMA,B_VAL]}
'''
#FIXME: Currently we are generating random values of GAMMA & B_VAL as the entire pipeline is not set and connected.

import csv
from collections import defaultdict
import random

agents_record = defaultdict(list)
with open('Participant_Details.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        if counter == 0:
            counter+=1
            continue
        agents_record[row['ETH_ADDRESS']] = [random.random(),random.random()]

#DRIVER
if __name__ == '__main__':
    print(agents_record)
