from web3 import Web3
from DeployContract import contract_instance,w3,contract_address,contract_private_key
from all_clients import all_clients
from whitelisted import whitelisted

#FIXME: Event Filters and Listeners as and when the updates are received.

w3.eth.defaultAccount = contract_instance.address

def setIpfsHash(yourHash):
	wallet_address = Web3.toChecksumAddress(all_clients['A'][0])
	nonce = w3.eth.getTransactionCount(wallet_address)

	txn_dict = contract_instance.functions.setIpfsHash(yourHash).buildTransaction({'chainId': 3, 'nonce': nonce,})

	signed_txn = w3.eth.account.signTransaction(txn_dict,private_key = all_clients['A'][1])
	result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

	okay = False
	while not okay:
		try:
			tx_receipt = w3.eth.getTransactionReceipt(result)
			okay = True
			print("YESS")
			print(tx_receipt)

		except:
			continue

#called to store sampled whitelisted agents on the smart contract.
#ARGUMENT - WHITELISTED AGENTS DICTIONARY (as per the format specified in the README)
def storeWhitelisted(my_dict):
	final_array = []
	for items in my_dict.keys():
		items = Web3.toChecksumAddress(items)
		final_array.append(items)

	nonce = w3.eth.getTransactionCount(contract_address)
	txn_dict = contract_instance.functions.whiteExperiment(final_array).buildTransaction({'from':contract_address,'chainId': 3,'nonce':nonce,})
	signed_txn = w3.eth.account.signTransaction(txn_dict,private_key = contract_private_key)
	result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

	count = 0
	okay = False
	while not okay:
		try:
			tx_receipt = w3.eth.getTransactionReceipt(result)
			okay = True
			print("Congratulations! Whitelisted agents stored on the blockchain\n")
			#print(tx_receipt)
		except:
			if count == 100000:
				print("There might be some error! CONTACT VIVEK!")
				break
			count+=1
			continue

#used to call a function in the contract that creates a mapping(address=>bool) in the smart contract. The mapping is used in the modifier logic and required.
#ARGUMENT: None
def whiteListMapping():
	nonce = w3.eth.getTransactionCount(contract_address)
	txn_dict = contract_instance.functions.createMapping().buildTransaction({'chainId': 3,'nonce':nonce,})
	signed_txn = w3.eth.account.signTransaction(txn_dict,private_key = contract_private_key)
	result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

	count = 0
	okay = False
	while not okay:
		try:
			tx_receipt = w3.eth.getTransactionReceipt(result)
			okay = True
			print("Congratulations! Mapping created successfully!\n")
			#print(tx_receipt)
		except:
			if count == 100000:
				print("There might be some error! CONTACT VIVEK!")
				break
			count+=1
			continue


#driver
if __name__ == '__main__':
	print("Connected to Ropsten:",str(w3.isConnected()))
	print("Server Account Address:",str(w3.eth.defaultAccount))
	balance = w3.eth.getBalance(contract_address)
	print("Server Account Balance (ETH):",str(w3.fromWei(balance,"ether")))
	#setIpfsHash("accdvdcdsbfsvvsfbfbaffsvsvsv")
	print("Adding Whitelisted agents to the Blockchain...")
	#populateWhiteListed(whitelisted)
	#whiteListTest(sampled)
	storeWhitelisted(whitelisted)
	whiteListMapping()
