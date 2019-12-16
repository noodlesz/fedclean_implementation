result_1 = {0:57.98,1:59.04,5:57.91,10:54.19,15:51.72,20:43.78,25:44.37,30:41.25}
result_2 = {0:57.98,1:58.04,5:56.12,10:51.83,15:47.44,20:40.62,25:40.92,30:35.76}
result_3 = {0:55.23,1:59.22,5:53.74,10:48.62,15:43.71,20:43.44,25:39.59,30:38.77}
result_4 = {0:57.82,1:56.65,5:54.63,10:50.74,15:45.25,20:42.01,25:36.78,30:38.28}
result_5 = {0:59.04,1:59.62,5:55.15,10:56.36,15:49.14,20:46.91,25:46.57,30:47.05}

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
	print(average_result)
