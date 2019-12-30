#DATA STRUCTURE: HASHMAP => KEYS are ACCOUNT_ADDRESSES, VALUES is a list in the following manner: ["PRIVATE_KEY","REP_SCORE(if needed)"]

#NOTE: THESE ARE NOT WHITELISTED CLIENTS. A SCRIPT NEEDS TO BE RUN BY ABHISHEK TO WHITELIST CLIENTS FROM THESE.

import csv
all_clients = {}

with open('Participant_Details.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    counter = 0
    for row in reader:
        if counter == 0:
            counter+=1
            continue
        all_clients[row['ETH_ADDRESS']] = row['PRIVATE_KEY']

#DRIVER
if __name__ == '__main__':
    print(all_clients)
