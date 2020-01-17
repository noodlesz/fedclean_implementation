import csv

with open("final_biasing_script.py","w+") as csv_file:
    writer = csv.writer(csv_file,delimiter=',')
    writer.writerow(("DIGIT","ORIGINAL_DIGIT_FREQ","0_ATTACK","5_ATTACK","10_ATTACK","15_ATTACK","20_ATTACK","25_ATTACK","30_ATTACK"))
    
    with open("biasing_prediction_freq_data.csv","r") as read_file:
        reader = csv.reader(read_file,delimiter=',')
        for rows in reader:
            print(rows)
