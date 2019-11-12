//SERVER uploads an array of S encrypted messages (using the TRUSTED USERS private key). Which means each TRUSTED USER can decrypt one message.
//ENCRYPTED MESSAGES WILL BE UPLOADED AS AN ARRAY
pragma solidity ^0.5.12;
pragma experimental ABIEncoderV2;

contract FilterTraining{

    address public ethServer;
    string[] public encryptedMessages;

    //constructor - currently used to make server the owner of the contract
    constructor () public {
        ethServer = msg.sender;
    }

    modifier onlyServer(){
        require(msg.sender == ethServer);
        _;
    }

    function uploadMessages(string[] memory _incomingMessages) onlyServer public{
        encryptedMessages = _incomingMessages;
    }

    function getMessages() public view returns(string[] memory){
        return encryptedMessages;
    }

}
