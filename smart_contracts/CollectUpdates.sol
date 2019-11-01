pragma solidity ^0.4.22;

contract CollectUpdates { 
    
    //state variables 
    mapping(address => Agent) public _acceptedIpfsHash;
    address public ethServer;
    address[] public acceptedAgents;
    
    //local variables
    uint private sampleSize = 0;
    bool private updatesCompleted = false;
    uint private receivedCount = 0;
    mapping(address => bool) private whiteListedAgents;
    
    
    //details of every agent is represented using a Struct.
    //address - ethereum address of every agent.
    //ipfsUpdatesHash - the hash address of agent's local updates stored on ipfs as a vector/array.
    struct Agent{
        address agent;
        string ipfsUpdatesHash;
    }
    
    //triggered when an update is received from an individual agent.
    event updateReceived(
        address agent,
        string ipfsUpdatesHash
    );
    
    //triggered when all the agents have submitted the updates.
    event allUpdatesReceived(
        bool updatesCompleted
    );

    //constructor - currently used to make server the owner of the contract
    constructor () public {
        ethServer = msg.sender;
    }
    
    modifier onlyServer(){
        require(msg.sender == ethServer);
        _;
    }
    
    modifier whiteListed(){
        require(whiteListedAgents[msg.sender] == true);
        _;
    }
    
    //converts bytes to address. used for string to address conversion
    function bytesToAddress (bytes memory b) public pure returns (address) {
        uint result = 0;
        for (uint i = 0; i < b.length; i++) {
            uint c = uint(b[i]);
            if (c >= 48 && c <= 57) {
                result = result * 16 + (c - 48);
                }
            if(c >= 65 && c<= 90) {
                result = result * 16 + (c - 55);
            }
            if(c >= 97 && c<= 122) {
                result = result * 16 + (c - 87);
            }
        }
        return address(result);
    }
    
    //used to set the sampleSize for the particular iteration and can only be called by the server. 
    function setSampleSize(uint _size) onlyServer public{
            if (sampleSize != 0){
                sampleSize = _size;
                    
            }
        }
    

    //once the server returns the address of sampled agents in an array, a loop will be run outside this contract and,
    //the sampled agents stored in the external array will be aded to acceptedAgents one by one.
    //NOTE: The contract needs addresses to be stored as BYTES20 value to be compatible with the address data types
    function addAccepted(string memory _newAddress) onlyServer public{
            bytes memory temp;
            temp = bytes(_newAddress);
            acceptedAgents.push(bytesToAddress(temp));
            whiteListedAgents[bytesToAddress(temp)] = true;
    }


    //a function to store the ipfsHash of the agents if they are whitelisted.
    //CALLED by agents themselves.
    //LOCAL UPDATES are serialized and added to IPFS externally. Only the hashes are stored on blockchain.
    function setIpfsHash(string memory _ipfsHash) whiteListed public{
        if (updatesCompleted == false){
            _acceptedIpfsHash[msg.sender] = Agent(msg.sender, _ipfsHash);
            receivedCount ++;
            emit updateReceived(msg.sender,_ipfsHash);
            
            if (receivedCount == sampleSize){
                updatesCompleted = true;
                emit allUpdatesReceived(updatesCompleted);
            }
        }
    }

}
