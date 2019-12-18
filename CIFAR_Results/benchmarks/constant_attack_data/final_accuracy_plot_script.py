result_1 = {0:86.96,1:84.79,5:73.57,10:57.27,15:49.04,20:39.45,25:40.87,30:40.09}
result_2 = {0:87.13,1:84.99,5:66.40,10:59.58,15:54.95,20:52.39,25:46.19,30:40.68}
result_3 = {0:86.68,1:85.24,5:73.67,10:57.72,15:61.59,20:47.58,25:48.12,30:45.94}
result_4 = {0:86.71,1:85.76,5:64.32,10:59.56,15:41.43,20:56.47,25:40.12,30:42.97}
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
