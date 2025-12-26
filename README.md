Cloud Firewall and Intrusion Detection Using Virtual Appliances

Author: Akerke Shynarbek
Program: MSc – Comprehensive Information Security
University: Satbayev University
Teacher: Abdul Razaque
Year: 2025

Project Overview

This project presents the development of a Hybrid Intrusion Detection System designed for cloud and virtualized environments. The system integrates rule-based traffic inspection with machine learning–based anomaly detection in order to provide accurate and efficient real-time threat detection. The proposed IDS operates as a virtual appliance and is capable of detecting reconnaissance activities such as port scanning, service enumeration, and abnormal TCP behavior.

The motivation behind the project is the limitation of traditional signature-based firewalls in detecting unknown or modified attacks. To address this problem, the system combines fast rule-based filtering with a Random Forest classifier trained on real network traffic data. The hybrid approach allows the system to maintain low latency while achieving high detection accuracy.

System Architecture

The system is divided into two main environments: the training environment and the deployment environment. The training environment is responsible for preparing the machine learning model, while the deployment environment handles live traffic monitoring and detection.

The training process is implemented in the file train_research_models.py. This script loads the CICIDS-2017 PortScan dataset, performs data cleaning and preprocessing, selects the most relevant network features, and trains two models: a baseline Decision Tree and an optimized Random Forest classifier. After evaluation, the optimized model is serialized and saved as optimized_ids_model.pkl for later use.

The deployment environment is implemented in hybrid_ids_final.py. This script acts as a virtual intrusion detection appliance. It captures live network traffic using Scapy, applies rule-based detection for known threats such as ICMP traffic, and forwards suspicious packets to the machine learning model for further analysis. The system outputs real-time alerts based on detection results.

Dataset and Features

The system uses the CICIDS-2017 dataset, specifically the PortScan subset. This dataset contains realistic network traffic and is widely used for intrusion detection research. Only four features were selected in order to maintain efficiency and ensure real-time performance. These features are destination port, flow duration, total forward packets, and total backward packets.

Traffic is labeled in binary form, where benign traffic is represented as zero and malicious traffic is represented as one. This allows the model to perform binary classification focused on reconnaissance detection.

Model Training and Evaluation

Model training is performed using train_research_models.py. Two classifiers are trained and compared. The first is a Decision Tree model used as a baseline. The second is a Random Forest model configured with one hundred estimators and optimized depth. The evaluation results show that the Random Forest model consistently outperforms the Decision Tree.

The baseline model achieved accuracy values between 95 and 96 percent, while the optimized Random Forest achieved accuracy between 98.1 and 98.2 percent. The Random Forest also demonstrated higher recall and lower false positive rates, making it more suitable for deployment in a real-time IDS.

After training, the optimized model is saved as optimized_ids_model.pkl and used by the live detection engine.

Live IDS Operation

The live intrusion detection system is executed using hybrid_ids_final.py. When started, the system loads the trained model and begins monitoring network traffic. Incoming packets are processed through two detection layers.

The first layer performs rule-based inspection and immediately detects known patterns such as ICMP traffic. This ensures fast response and minimal computational cost. The second layer applies machine learning–based detection by extracting traffic features and evaluating them using the trained Random Forest model.

If malicious behavior is detected, the system prints an alert containing the confidence score. This confirms that both detection layers operate correctly and simultaneously.

Live Testing and Results

Live testing was performed using common network reconnaissance tools. ICMP flooding was used to validate the rule-based detection layer. TCP stealth scanning and service enumeration were used to test the machine learning model. OS fingerprinting and SSH probing were also conducted to evaluate anomaly detection.

During these tests, the system successfully detected malicious behavior and generated alerts with confidence values ranging from 0.81 to 1.00. The IDS consistently differentiated between normal traffic and attack patterns, confirming the reliability of the hybrid approach.

Dataset Limitations

The CICIDS-2017 dataset represents traffic generated in a controlled environment and does not include all possible real-world attack scenarios. Only the PortScan subset was used, which limits the system to reconnaissance detection. Additionally, the use of a small number of features, while beneficial for performance, may reduce sensitivity to more complex attack behaviors. Encrypted traffic is also not explicitly handled in the current implementation.

Despite these limitations, the dataset was sufficient to validate the effectiveness of the proposed hybrid IDS and demonstrate accurate detection in real-time conditions.

Conclusion

This project successfully demonstrates the design and implementation of a hybrid intrusion detection system that combines rule-based filtering with machine learning–based analysis. The system achieves high detection accuracy while remaining lightweight and suitable for deployment in virtualized environments. The use of a Random Forest classifier significantly improves detection performance compared to a baseline model. Real-time testing confirms that the IDS can detect reconnaissance attacks reliably and efficiently.
