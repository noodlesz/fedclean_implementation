''' VARIABLES EXPECTED:
a) Trade-Off Parameter (Alpha)
b) Weight/Reputation Score (Gamma)
c) Last Time The Agent was selected (b)

RETURNS a LIST of addresses of SAMPLED AGENTS
'''
#agents_record = {"ETH_ADDRESS":[GAMMA,B_VAL]}
from dataForAgentSelection import agents_record
from collections import defaultdict,OrderedDict

def calc_sum(agents_record):
	sum_gamma = 0
	sum_b_val = 0
	for items in agents_record.keys():
		sum_gamma+=agents_record[items][0]
		sum_b_val+=agents_record[items][1]

	return sum_gamma,sum_b_val

def calc_probabilities(agents_record,trade_off_param):
	ret_mapping = defaultdict(int)
	sum_gamma,sum_b_val = calc_sum(agents_record)

	for items in agents_record.keys():
		agent_prob = (trade_off_param*(agents_record[items][0]/sum_gamma)) + ((1-trade_off_param)*(agents_record[items][1]/sum_b_val))
		ret_mapping[items] = agent_prob

	return ret_mapping

def sample_agents(number,final_structure):
	ret_list = []
	dd = OrderedDict(sorted(final_structure.items(), key = lambda x: x[1],reverse=True))
	dd = dict(dd)

	counter = 0
	for items in dd.keys():
		if counter == number:
			break
		ret_list.append(items)
		counter+=1
	return ret_list

##DRIVER##
if __name__ == '__main__':
	print("The Sampled Agents are:")
	#a_record = {"ascaadcadcac":[0.5,0.4],"ssacdcdac":[0.9,0.4],"adscdac":[0.8,0.9]}
	trade_off = 0.6
	final = calc_probabilities(agents_record,trade_off)
	print(sample_agents(6,final))
