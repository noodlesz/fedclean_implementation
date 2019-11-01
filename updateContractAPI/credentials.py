import json

with open("my_abi.json") as f:
	info_json = json.load(f)

req_abi = info_json["abi"]


#driver
if __name__ == "__main__":
	print(req_abi)
