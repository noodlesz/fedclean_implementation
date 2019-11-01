import ipfsApi
from web3 import Web3
from DeployContract import contract_instance,w3,contract_address,contract_private_key
from all_clients import all_clients
from whitelisted import whitelisted

w3.eth.defaultAccount = contract_instance.address

#requires installation of ipfs-daemon on your local computer
api = ipfsApi.Client('127.0.0.1', 5001)

def setIpfsHash(yourKey,yourHash):
	try:
		wallet_address = Web3.toChecksumAddress(yourKey)
		nonce = w3.eth.getTransactionCount(wallet_address)

		txn_dict = contract_instance.functions.setIpfsHash(yourHash).buildTransaction({'chainId': 3, 'nonce': nonce,})

		signed_txn = w3.eth.account.signTransaction(txn_dict,private_key = all_clients[yourKey][0])
		result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

	except KeyError as e:
		print("Maybe the key is incorrect or not whitelisted. Sorry.")
	
	count = 0
	okay = False
	while not okay:
		try:
			tx_receipt = w3.eth.getTransactionReceipt(result)
			okay = True
			print("Added for:",str(wallet_address),"\nHash Value:",str(yourHash))
			#print(tx_receipt)

		except:
			if count == 100000:
				print("There might be some error. Contact Vivek.")
				break
			count+=1
			continue

def addAll():
	for items in whitelisted.keys():
		ipfs_hash = api.add_pyobj(whitelisted[items])
		setIpfsHash(items,ipfs_hash)


#FIXME: CAN ONLY BE RUN AND TESTED WHEN WE HAVE AGENT SAMPLING SCRIPT AND WHITELISTED CLIENTS.

#these updates can be received back as PYTHON LISTS using api.get_pyobj(HASH). The HASH value for each agent will be found from the mapping stored on smart contract.

#driver
if __name__ == '__main__':
	print("Storing Hashes to Blockchains...")
	addAll()

