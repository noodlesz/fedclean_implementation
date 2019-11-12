'''
EXPECTED_INPUTS:
a) CURRENT_REPUTATION_SCORE
b) CONSTANT K
c) NO. of SAMPLED AGENTS (SAMPLE SIZE)
d) STRENGTH OF EACH REPORT
e) MAX_STRENGTH
f) SECOND_MAX_STRENGTH

RETURN:
a) NEW_REPUTATION
'''
max_strength = #constant
second_max_strength = #constant
def calculateReputation(incomingReputation,k,sample_size,report_strength):
    new_reputation = 0
    if incomingReputation == max_strength:
        new_reputation = (1/(k*sample_size))*(1-incomingReputation)*(max_strength-second_max_strength)

    elif incomingReputation < max_strength:
        new_reputation = (1/(k*sample_size)) * (incomingReputation)*(incomingReputation-max_strength)


    return new_reputation
