pragma solidity ^0.4.22;

contract CollectUpdates {


    //state variables
    mapping(address => Agent) public _acceptedIpfsHash;
    address public ethServer;
    address[] public acceptedAgents;

    //local variables
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


    function whiteExperiment(address[] memory _hopeForTheBest) onlyServer public{
        acceptedAgents = _hopeForTheBest;
    }

    function createMapping() onlyServer public{
        for (uint i = 0; i<acceptedAgents.length; i++){
            whiteListedAgents[acceptedAgents[i]] = true;
        }
    }


    //a function to store the ipfsHash of the agents if they are whitelisted.
    //CALLED by agents themselves.
    //LOCAL UPDATES are serialized and added to IPFS externally. Only the hashes are stored on blockchain.
    function setIpfsHash(string memory _ipfsHash) whiteListed public{
        if (updatesCompleted == false){
            _acceptedIpfsHash[msg.sender] = Agent(msg.sender, _ipfsHash);
            receivedCount ++;
            emit updateReceived(msg.sender,_ipfsHash);

            if (receivedCount == acceptedAgents.length){
                updatesCompleted = true;
                emit allUpdatesReceived(updatesCompleted);
            }
        }
    }

}
