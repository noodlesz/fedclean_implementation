import ipfsApi
from whitelisted import whitelisted

#requires installation of ipfs-daemon on your local computer
api = ifsApi.Client('127.0.0.1', 5001)

for items in whitelisted.keys():
	ipfs_hash = api.add_pyobj(whitelisted[items][1])




#FIXME: CAN ONLY BE RUN AND TESTED WHEN WE HAVE AGENT SAMPLING SCRIPT AND WHITELISTED CLIENTS.

#driver
#FIXME: TO BE ADDED


