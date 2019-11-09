from StoredHashes import hash_store
import ipfsApi


#requires installation of ipfs-daemon on your local computer
api = ipfsApi.Client('127.0.0.1', 5001)


all_local_updates = []
for items in hash_store.keys():
    curr_update = api.get_pyobj(hash_store[items])
    all_local_updates.append(curr_update)


##DRIVER##
if __name__ == '__main__':
    print(all_local_updates)
