//STORING THE REPUTATION SCORES ON THE SMART CONTRACT
//DATA STRUCTURE - MAPPING from ADDRESS => SCORES (STRINGS)

pragma solidity ^0.5.12;
pragma experimental ABIEncoderV2;

contract Reputation{

    address public ethServer;

    //the reputation scores will be received as a string from the python script as Solidity doesn't support float and we are also not using scores for any computations
    string[] public incomingReputation;
    address[] public whiteListedAgents;
    mapping(address=>string) public finalScoreMapping;

    //constructor - currently used to make server the owner of the contract
    constructor () public {
        ethServer = msg.sender;
    }

    //modifier that only allows the server to add updates
    modifier onlyServer(){
        require(msg.sender == ethServer);
        _;
    }

    //get OLD REPUTATION scores array from web3py
    function getRepScoreFromWeb3Py(string[] memory _inputReps) onlyServer public{
        incomingReputation = _inputReps;
    }

    //get an array of strenght of each reputation as determined by SVDD
    function getAgents(address[] memory _inputStrengths) onlyServer public{
        whiteListedAgents = _inputStrengths;
    }

    //create the mapping for received inputs
    function createMapping() onlyServer public{
        for(uint i = 0; i < incomingReputation.length; i++){
            finalScoreMapping[whiteListedAgents[i]] = incomingReputation[i];
        }
    }
}
