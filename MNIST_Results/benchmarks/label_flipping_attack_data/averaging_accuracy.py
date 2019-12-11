import csv 

overall_average_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_0_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_1_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_2_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_3_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_4_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_5_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_6_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_7_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_8_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}
overall_9_test_accuracy = {1:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},2:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},3:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},4:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0},5:{0:0,1:0,5:0,10:0,15:0,20:0,25:0,30:0}}

internal_counter = [0,1,5,10,15,20,25,30]
i = 0

with open("1_label_flipped_final_data.csv") as f:
	reader = csv.reader(f, delimiter=',')
	counter = 1
	for row in reader:
		if counter==1:
			counter+=1
			continue
		row_counter = 1
		#print(row)
		counter+=1

		#print(type(row),type(overall_average_test_accuracy),type(internal_counter))
		overall_average_test_accuracy[1][internal_counter[i]] = float(row[1])
		overall_0_test_accuracy[1][internal_counter[i]] = float(row[3])
		overall_1_test_accuracy[1][internal_counter[i]] = float(row[5])
		overall_2_test_accuracy[1][internal_counter[i]] = float(row[7])
		overall_3_test_accuracy[1][internal_counter[i]] = float(row[9])
		overall_4_test_accuracy[1][internal_counter[i]] = float(row[11])
		overall_5_test_accuracy[1][internal_counter[i]] = float(row[13])
		overall_6_test_accuracy[1][internal_counter[i]] = float(row[15])
		overall_7_test_accuracy[1][internal_counter[i]] = float(row[17])
		overall_8_test_accuracy[1][internal_counter[i]] = float(row[19])
		overall_9_test_accuracy[1][internal_counter[i]] = float(row[21])

		i += 1

i = 0

with open("2_label_flipped_final_data.csv") as f:
	reader = csv.reader(f, delimiter=',')
	counter = 1
	for row in reader:
		if counter==1:
			counter+=1
			continue
		row_counter = 1
		#print(row)
		counter+=1

			
		overall_average_test_accuracy[2][internal_counter[i]] = float(row[1])
		overall_0_test_accuracy[2][internal_counter[i]] = float(row[3])
		overall_1_test_accuracy[2][internal_counter[i]] = float(row[5])
		overall_2_test_accuracy[2][internal_counter[i]] = float(row[7])
		overall_3_test_accuracy[2][internal_counter[i]] = float(row[9])
		overall_4_test_accuracy[2][internal_counter[i]] = float(row[11])
		overall_5_test_accuracy[2][internal_counter[i]] = float(row[13])
		overall_6_test_accuracy[2][internal_counter[i]] = float(row[15])
		overall_7_test_accuracy[2][internal_counter[i]] = float(row[17])
		overall_8_test_accuracy[2][internal_counter[i]] = float(row[19])
		overall_9_test_accuracy[2][internal_counter[i]] = float(row[21])

		i += 1

i = 0

with open("3_label_flipped_final_data.csv") as f:
	reader = csv.reader(f, delimiter=',')
	counter = 1
	for row in reader:
		if counter==1:
			counter+=1
			continue
		row_counter = 1
		counter+=1

			
		overall_average_test_accuracy[3][internal_counter[i]] = float(row[1])
		overall_0_test_accuracy[3][internal_counter[i]] = float(row[3])
		overall_1_test_accuracy[3][internal_counter[i]] = float(row[5])
		overall_2_test_accuracy[3][internal_counter[i]] = float(row[7])
		overall_3_test_accuracy[3][internal_counter[i]] = float(row[9])
		overall_4_test_accuracy[3][internal_counter[i]] = float(row[11])
		overall_5_test_accuracy[3][internal_counter[i]] = float(row[13])
		overall_6_test_accuracy[3][internal_counter[i]] = float(row[15])
		overall_7_test_accuracy[3][internal_counter[i]] = float(row[17])
		overall_8_test_accuracy[3][internal_counter[i]] = float(row[19])
		overall_9_test_accuracy[3][internal_counter[i]] = float(row[21])

		i += 1

i = 0

with open("4_label_flipped_final_data.csv") as f:
	reader = csv.reader(f, delimiter=',')
	counter = 1
	for row in reader:
		if counter==1:
			counter+=1
			continue
		row_counter = 1
		counter+=1

			
		overall_average_test_accuracy[4][internal_counter[i]] = float(row[1])
		overall_0_test_accuracy[4][internal_counter[i]] = float(row[3])
		overall_1_test_accuracy[4][internal_counter[i]] = float(row[5])
		overall_2_test_accuracy[4][internal_counter[i]] = float(row[7])
		overall_3_test_accuracy[4][internal_counter[i]] = float(row[9])
		overall_4_test_accuracy[4][internal_counter[i]] = float(row[11])
		overall_5_test_accuracy[4][internal_counter[i]] = float(row[13])
		overall_6_test_accuracy[4][internal_counter[i]] = float(row[15])
		overall_7_test_accuracy[4][internal_counter[i]] = float(row[17])
		overall_8_test_accuracy[4][internal_counter[i]] = float(row[19])
		overall_9_test_accuracy[4][internal_counter[i]] = float(row[21])

		i += 1


i = 0

with open("5_label_flipped_final_data.csv") as f:
	reader = csv.reader(f, delimiter=',')
	counter = 1
	for row in reader:
		if counter==1:
			counter+=1
			continue
		row_counter = 1
		counter+=1

			
		overall_average_test_accuracy[5][internal_counter[i]] = float(row[1])
		overall_0_test_accuracy[5][internal_counter[i]] = float(row[3])
		overall_1_test_accuracy[5][internal_counter[i]] = float(row[5])
		overall_2_test_accuracy[5][internal_counter[i]] = float(row[7])
		overall_3_test_accuracy[5][internal_counter[i]] = float(row[9])
		overall_4_test_accuracy[5][internal_counter[i]] = float(row[11])
		overall_5_test_accuracy[5][internal_counter[i]] = float(row[13])
		overall_6_test_accuracy[5][internal_counter[i]] = float(row[15])
		overall_7_test_accuracy[5][internal_counter[i]] = float(row[17])
		overall_8_test_accuracy[5][internal_counter[i]] = float(row[19])
		overall_9_test_accuracy[5][internal_counter[i]] = float(row[21])

		i += 1




def findAverage(my_dict):
	#ret_tuple = (attack_0,attack_1,attack_5,attack_10,attack_15,attack_20,attack_25,attack_30)
	attack_0_sum,attack_1_sum,attack_5_sum,attack_10_sum,attack_15_sum,attack_20_sum,attack_25_sum,attack_30_sum = 0,0,0,0,0,0,0,0
	for run_val in range(1,6):
		attack_0_sum += my_dict[run_val][0]
		attack_1_sum += my_dict[run_val][1]
		attack_5_sum += my_dict[run_val][5]
		attack_10_sum += my_dict[run_val][10]
		attack_15_sum += my_dict[run_val][15]
		attack_20_sum += my_dict[run_val][20]
		attack_25_sum += my_dict[run_val][25]
		attack_30_sum += my_dict[run_val][30]

	return (attack_0_sum/5),(attack_1_sum/5),(attack_5_sum/5),(attack_10_sum/5),(attack_15_sum/5),(attack_20_sum/5),(attack_25_sum/5),(attack_30_sum/5)

overall_test_average = findAverage(overall_average_test_accuracy)
overall_0_average = findAverage(overall_0_test_accuracy)
overall_1_average = findAverage(overall_1_test_accuracy)
overall_2_average = findAverage(overall_2_test_accuracy)
overall_3_average = findAverage(overall_3_test_accuracy)
overall_4_average = findAverage(overall_4_test_accuracy)
overall_5_average = findAverage(overall_5_test_accuracy)
overall_6_average = findAverage(overall_6_test_accuracy)
overall_7_average = findAverage(overall_7_test_accuracy)
overall_8_average = findAverage(overall_8_test_accuracy)
overall_9_average = findAverage(overall_9_test_accuracy)



##DRIVER##
if __name__ == '__main__':
	print("Overall",str(overall_test_average))
	print("0 average",str(overall_0_average))
	print("1 average",str(overall_1_average))
	print("2 average",str(overall_2_average))
	print("3 average",str(overall_3_average))
	print("4 average",str(overall_4_average))
	print("5 average",str(overall_5_average))
	print("6 average",str(overall_6_average))
	print("7 average",str(overall_7_average))
	print("8 average",str(overall_8_average))
	print("9 average",str(overall_9_average))



