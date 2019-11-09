import ipfsApi
from web3 import Web3
from DeployStateContract import contract_instance,w3,contract_address,contract_private_key

w3.eth.defaultAccount = contract_instance.address

#requires installation of ipfs-daemon on your local computer
api = ipfsApi.Client('127.0.0.1', 5001)

def updateStateOnContract(global_update_array):
    #Uploading on ipfs
    ipfs_hash = api.add_pyobj(global_update_array)


    nonce = w3.eth.getTransactionCount(contract_address)
    gas_price = w3.eth.gasPrice

    txn_dict = contract_instance.functions.setState(ipfs_hash).buildTransaction({'from':contract_address,'chainId': 3, 'nonce': nonce,'gasPrice':gas_price,})

    signed_txn = w3.eth.account.signTransaction(txn_dict,private_key = contract_private_key)
    result = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

    count = 0
    okay = False
    while not okay:
        try:
            tx_receipt = w3.eth.getTransactionReceipt(result)
            okay = True
            print("FUCKING INSAE")
            print(ipfs_hash)
    		#print("Added for:",str(wallet_address),"\nHash Value:",str(yourHash))
    		#print(tx_receipt)

        except:
            if count == 100000:
                print("There might be some error. Contact Vivek.")
                break
            count+=1
            continue

##DRIVER##
if __name__ == '__main__':
    updateStateOnContract([0.005,0.989,0.345])
