from credentials import state_abi
from web3 import Web3

contract_address = Web3.toChecksumAddress('0xd440d92dc5357d297b6232f2c8bb1e6911483b69')
contract_private_key = "E34F12503E4A84100E86A920B5C69D8FBFF80AC9C8293CA974374514A7676EF8"

provider = "https://ropsten.infura.io/v3/dcb7743a98a142af812ed7b849137ec4"
w3 = Web3(Web3.HTTPProvider(provider))

contract_instance =	w3.eth.contract(address=contract_address,abi = state_abi)

#driver
if __name__ == '__main__':
	try:
		print(w3.isConnected())
		print("Deployment Successful. Contract deployed on Ropsten TestNet using Infura API.")
		print("\t\tAddress: ",str(contract_instance.address))
		print("\t\tMethods: ",str(contract_instance.functions))
		print("\t\tEvents:  ",str(contract_instance.events))
		print("\t\tABI:     ",str(state_abi))

	except:
		print("Error creating contract instance. Contact Vivek.")
