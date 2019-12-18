#FIRST FOUR are the BEST ONES :D

result_1 = {0:93.52,1:93.59,5:91.78,10:82.71,15:80.43,20:40.68,25:42.78,30:28.87}
result_2 = {0:87.08,1:76.35,5:72.89,10:59.40,15:56.49,20:48.12,25:45.03,30:36.23}
result_3 = {0:94.89,1:92.18,5:79.30,10:67.42,15:51.58,20:49.96,25:40.35,30:40.90}
result_4 = {0:91.26,1:85.14,5:61.88,10:44.55,15:53.12,20:44.90,25:45.76,30:41.52}
result_5 = {0:86.90,1:81.81,5:64.72,10:51.99,15:50.51,20:40.79,25:35.87,30:36.03}



attack_none = (result_1[0] + result_2[0] + result_3[0] + result_4[0] + result_5[0])/5
attack_1 = (result_1[1] + result_2[1] + result_3[1] + result_4[1] + result_5[1])/5
attacK_5 = (result_1[5] + result_2[5] + result_3[5] + result_4[5] + result_5[5])/5
attack_10 = (result_1[10] + result_2[10] + result_3[10] + result_4[10] + result_5[10])/5
attack_15 = (result_1[15] + result_2[15] + result_3[15] + result_4[15] + result_5[15])/5
attack_20 = (result_1[20] + result_2[20] + result_3[20] + result_4[20] + result_5[20])/5
attack_25 = (result_1[25] + result_2[25] + result_3[25] + result_4[25] + result_5[25])/5
attack_30 = (result_1[30] + result_2[30] + result_3[30] + result_4[30] + result_5[30])/5

average_result = {0:attack_none,1:attack_1,5:attacK_5,10:attack_10,15:attack_15,20:attack_20,25:attack_25,30:attack_30}

##DRIVER##
if __name__ == '__main__':
	for items in average_result.keys():
		print(items,str(average_result[items]))
