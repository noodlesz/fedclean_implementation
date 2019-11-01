from web3 import Web3
from DeployContract import contract_instance,w3,contract_address,contract_private_key
from all_clients import all_clients
from whitelisted import whitelisted

#FIXME: Event Filters and Listeners as and when the updates are received. 

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
			print(tx_receipt)

		except:
			continue

#FIXME: TEST IT when there is some actual WHITELISTED data.
#FIXME: APPENDING ONE ACCOUNT in a single transaction takes time but still works. Try to optimize it by trying to upload whole array.

def populateWhiteListed(final_dict):
	for items in final_dict.keys():
		items = Web3.toChecksumAddress(items)
		nonce = w3.eth.getTransactionCount(contract_address)
		txn_dict = contract_instance.functions.addAccepted(items).buildTransaction({'chainId': 3, 'nonce':nonce,})
		signed_txn = w3.eth.account.signTransaction(txn_dict, private_key =	contract_private_key)
		result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
		
		count = 0
		okay = False
		while not okay:
			try:
				tx_receipt = w3.eth.getTransactionReceipt(result)
				okay = True
				print("Added:",str(items))
				check_list.append(str(items))

			except:
				if count == 100000:
					print("There might be some error. Check your addresses or contact Vivek.")
					break
				count+=1
				continue

			
#driver
if __name__ == '__main__':
	print("Connected to Ropsten:",str(w3.isConnected()))
	print("Server Account Address:",str(w3.eth.defaultAccount))
	balance = w3.eth.getBalance(contract_address)
	print("Server Account Balance (ETH):",str(w3.fromWei(balance,"ether")))
	#setSampleSize(10)
	#setIpfsHash("accdvdcdsbfsvvsfbfbaffsvsvsv")
	print("Adding Whitelisted agents to the Blockchain...")
	populateWhiteListed(whitelisted)
