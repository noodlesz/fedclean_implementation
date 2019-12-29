- **Author:** Vivek Khimani
- **Contact:** +12673049080
- **Purpose:** Explaining the CSV file format and experimentation data for plot generation.
- **Target Collaborator:** Dr. Dimitris Chatzopoulos
- **Concerned Files:** constant_attack_plot_data_3.csv, accuracy_plot_data.csv

---

## Files Content Summary:

- **constant_attack_plot_data_3.csv** -> Contains the log of training loss for every epoch (total of 50 epochs), for federated learning system containing different number of malicious agents. More information will be given in Files Structure section, but ideally the training loss should be more for the system containing increased number of malicious agents. 
- **accuracy_plot_data.csv** -> Contains the average testing accuracy (5 runs) for different federated learning systems containing different number of malicious agents. Testing accuracy for the systems with more malicious agents should decrease in the ideal scenario.

## Files Structure:

- **constant_attack_plot_data_3.csv** -> columns: represent federated learning systems with different number of malicious agents; rows: represent training loss for each system in 1 epoch (range: [1:50])

| 0_malicious_agents    | 1_malicious_agents   | 5_malicious_agents    | 10_malicious_agents    | 15_malicious_agents    | 20_malicious_agents    | 25_malicious_agents    | 30_malicious_agents |
## Intended Graphs:

- **Figure 1** on FedClean Overleaf, i.e. Training Loss of Federated Learning System in presence of 0 to 30 malicious agents over 50 epochs of training. 
- **Figure 2** on FedClean Overleaf, i.e. Testing Accuracy in presence of 0 to 30 agents after 50 rounds of training in 2 setting. 1) Natural (no defence) federated learning 2) FedClean (defence).

** NOTE ** We don't have any data for FedClean accuracy for now, so just make a bar graph for one. We will add a combined comparison graph once we have a data for both. 