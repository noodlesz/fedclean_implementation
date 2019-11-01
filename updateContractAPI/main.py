from web3 import Web3
from DeployContract import contract_instance,w3,contract_address,contract_private_key
from all_clients import all_clients

w3.eth.defaultAccount = contract_instance.address

def setSampleSize(count):
	nonce = w3.eth.getTransactionCount(contract_address)

	txn_dict = contract_instance.functions.setSampleSize(count).buildTransaction({'chainId': 3, 'nonce':nonce,})
	signed_txn = w3.eth.account.signTransaction(txn_dict, private_key = contract_private_key)
	result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
	
	okay = False
	while not okay:
		try:
			tx_receipt = w3.eth.getTransactionReceipt(result)
			okay = True
			print(tx_receipt)
			print("SUCCESS")

		except:
			continue

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
			procc = contract_instance.events.updateReceived(wallet_address,yourHash)
			print(procc)

		except:
			continue



#driver
if __name__ == '__main__':
	print(w3.isConnected())
	print(w3.eth.defaultAccount)
	print(w3.eth.blockNumber)
	balance = w3.eth.getBalance(contract_address)
	print(w3.fromWei(balance,"ether"))
	#setSampleSize(10)
	setIpfsHash("accdvdcdsbfsvvsfbfbaffsvsvsv")
