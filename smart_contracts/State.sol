pragma solidity ^0.5.12;

contract State{

    address public ethServer;
    string public stateIpfsHash;



    constructor() public{
        ethServer = msg.sender;
    }


    modifier onlyServer(){
        require(msg.sender == ethServer);
        _;
    }

    function setState(string memory _globalIpfsHash) onlyServer public{
        stateIpfsHash = _globalIpfsHash;
    }

    function getState() public view returns(string memory){
        return stateIpfsHash;
    }

}
