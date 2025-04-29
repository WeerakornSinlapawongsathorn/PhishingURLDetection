# PhishingURLDetection

# ****Abstract****

Phishing attacks have become a major threat because it can trick users to reveal their sensitive information such as username, password, financial information etc. Phishing URLs can be attached in emails containing fake or attract information. This could easily lure victims to click and expose their confidentiality. Phishing URLs are increasing, this large amount of phishing is difficult to detect. Machine Learning (ML) is an alternative method to handle large datasets and provide high efficiency for phishing detection. In this study, 70% and 30% of over 200,000 datasets containing over 100,945 phishing URLs were trained and tested by Logistic Regression and Support Vector Machine (SVM) models. Since phishing URLs have specific characteristics, 9 features were extracted from URLs as an input, legitimate URLs are defined as 1 and phishing URLs are labeled to 1. The label data was treated as a target value. These two models’ performance was measured by accuracy, precision, recall, and f1. As a result, both models were effective to detect phishing URLs at over 80% accuracy rate, whereas SVM presented slightly higher performance. Nevertheless, the previous studies provided very high efficiency at over 90% accuracy with more complex and larger feature extraction. This proposed study suggested that feature extraction significantly affects ML’s performance; more complicated extraction could carry a higher accuracy rate.


# **1. Background**

Social Engineering is a popular cybercrime to take advantage of human errors to steal personal or sensitive information. Phishing is one mechanism of social engineering to lure victims by adapting technical tricks such as spoofed emails, promotional emails, illegitimate websites or social platform attached with suspicious links. These links or URLs contain suspicious characters, have long-string or use insecure protocol such as Hypertext Transport Protocol (HTTP). The phishers also conduct URLs with or without IP addresses. One-click on those links redirect users to suspicious webpage; an attacker takes this opportunity to steal personal information, including usernames, passwords and financial data. Moreover, the links attached with malware impact on computers or IT devices. 

Regarding those impacts, cybersecurity is important to prevent this type of social engineering. Firewalls are prominent to set up a whitelist for accepted network traffic. Some companies provide security policies to allow only secure networks or protocols such as HTTPS, SSH etc. Nevertheless, artificial intelligence (AI) becomes prevalent. The phishers perform AI generating phishing URLs or emails. DeepPhish is a malicious AI to generate AI Phishing URLs [1]. In 2017, the Anti-Phishing Working Group (APWG) reported 25% of phishing URLs under HTTPS protocol, this rose up to 83% in 2021[2]. APWG revealed 1,270,883 unique phishing attacks in 2022, higher than past several years [3]. This rising cybercrime indicated a requirement of strong detection and prevention. The increasing phishing cyberattacks enlarge the amount of data collected for detection. Machine Learning (ML) is beneficial to rely on large data and learn to classify the characteristics of phishing attacks. The proposed study utilized ML models, Logistic Regression and Support Vector Machine (SVM), to classify the phishing URLs.

# **2. Statemente of the problem**

As an increasing phishing attack amplifies the amount of data for detection and prevention. This consumes high storage and produces complexity for current resources such as firewalls or log management. Supervised ML is an efficient method to simplify classification and detection. 

# **3.  Objectives**

-	To utilize ML algorithms for phishing URLs classification and detection
-	To study the effectiveness of Logistic Regression and SVM algorithms

# **4. Methodology Overview**

The proposed study downloaded an available dataset form https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset, which contains 134,850 legitimate and 100,945 phishing URLs. The dataset is provided in the CVS file. The phishing URLs are labeled as 0, whereas the legitimate URLs are designated as 1. The URL data and label data were extracted as an input feature and a target for detection. Logistic Regression and SVM were utilized. As Logistic Regression is suitable for binary classification using Sigmoid function to map between 0 and 1, the cost function is used to measure different between a target value and predicted value. SVM aims to optimize hyperplane to separate two classes, which was phishing and legitimate URLs. SVM uses hinge loss to maximize the decision boundary between classes. These models’ performance was measured by accuracy, precision, recall, and f1.

# **5. Implementation**

This study utilized Python programming language, which includes necessary libraries for ML such as Sklearn, and Pandas. The characteristics of phishing emails, which consists of special characters (e.g., @, #, and ‘?’), strange subdomain, unusual length, number of digits etc. IP of each URL was investigated to check if the URLs exist. Phishing URLs usually use both HTTP and HTTPs for communication protocols. In addition, the phishing URLs are often shortened to trick victims as a legitimate URL such as “bit.ly” or contain high entropy. These specific characteristics were used to extract the characters in URLs in the dataset in this study. 
After the feature extraction process has been completed, these features created a DataFrame containing the extracted features and labels for phishing and legitimate URLs. The label column was set as a target. 70% of this data was trained and the rest was tested by ‘train_test_split()’ functoin. Due to the large dataset, StandardScaler was applied to set mean and standard deviation as 0 and 1 of the training and testing data, reducing complexity and increasing efficiency.  
Logistic Regression and SVM models trained and tested 70% and 30% of total data. Logistic Regression and SVM inherently provide their own cost functions, which are used to evaluate the gap between actual and predictable values. Logistic Regression predicted and classified the data in either 0 or 1, which was the label for phishing and legitimate. Meanwhile, SVM has both 1 and -1, which presented legitimate and phishing values. SVM applies hinge loss to optimize the function and maximize hyperplane between two classes. True positive rate was mainly focused on. Hence, the performance was measured by accuracy rate, precision, recall and f1_score based on True Positive Rate (TP), True Negative Rate (TN), False Positive Rate (FP), and False Negative Rate (FN) [1] in confusion matrix. The results of each model were comparable. 


![image](https://github.com/user-attachments/assets/c4377f43-a9e4-4f01-9fc2-9424e52c358d)

Figure 2.1 Methodology Flow

# ****Results and Discussion****

## **Logistic Regression Model**

Logistic Regression illustrated over 80% of its performance as shown in Table 3.1. Figure 3.1 presents the confusion matrix, 19,802 phishing URLs were detected correctly, but 468 legitimate URLs were exposed as phishing URLs, resulting in 0.343 false positive rate. Additionally, 10,349 phishing URLs were misclassified as legitimate URLs, leading to 0.012 false negative rate. The precision of phishing URLs detection was 87% while the recall was 84%.  However, the results were low when compared to [1], which reported over 90% phishing detection. 

[1] proposed PhishHaven, a real-time Phishing URLs detection designed to encounter DeepPhish. PhishHaven could be attributed to unique features, which emphasized greater special characters and URL HTML Encoding to differentiate between AI-generated Phishing URLs and Common Phishing URLs. Furthermore, [1] extracted URLs into types of protocol, netloc, path, query, and fragment. This extraction provided more structure for higher performance.

In comparison, [4] applied Logistic Regression to detect Phishing URLs from PhiUSIIL dataset, which aligns with the dataset in this proposed study. In contrast, [4] experimented on several features: 5, 10, 15, 20 and 25 features. The 5 features in [4] consisted of URLSimilarityIndex, LineOfCode, NoOfExternalRef, NoOfImage, and NoOfSelfRef. Despite having fewer features, [1] and [4] achieved over 90% phishing detection accuracy. Throughout these studies, the choice and complexity of features significantly impact on model performance.

![image](https://github.com/user-attachments/assets/289d10b6-82fd-438a-b317-60d44aaf3cf4)

Figure 3.1 Confusion Matrix of Logistic Regression

## **Support Vector Machine Model**

SVM resulted in over 80% Phishing URLs detection, which was slightly higher than Logistic Regression (Table 3.1). The reason is Logistic Regression works best for structured data and binary classification; SVM supports unstructured data and optimizes hyperplanes for ML. Figure 3.2 depicted the correctly detected 20,347 phishing URLs, but 9,804 phishing URLs were miscategorized as legitimate URLs. Nonetheless, its performance was lower than [5].

[5] detected phishing from 10,000, 15,000, 20,000, 25,000, and 30,000 datasets by using SVM algorithm. A total of 18 features extracted from URLs almost aligned with this study, but [5] added an extraction of port number, Unicode characters, redirection, suspicious symbols, and presence of ‘www’. [5] achieved over 90% accuracy and reported that the larger features depicted higher performance of SVM. However, even though [5] extracted larger features, the dataset was lower than this proposed study, which might affect the results.  

![image](https://github.com/user-attachments/assets/e85250b9-4f67-4806-a602-241d25e47c10)
Figure 3.2 Confusion Matrix of Support Vector Machine

|	        | Logistic Regression | SVM
|-------- |---------------------|------
|Accuracy |	0.847	              |0.856
|Precision|	0.873	              |0.879
|Recall	  | 0.847               |0.856
|F1_Score	| 0.840	              |0.850
|TP	      | 0.988	              |0.992
|TN	      | 0.657	              |0.670
|FP	      | 0.343	              |0.330
|FN	      |0.012	              |0.008

# ****Conclusion****

Both Logistic Regression and SVM performed over 80% accuracy for phishing URL detection. SVM was slightly greater. Logistic Regression indicated 0.987 TP, whereas SVM presented 0.990 TP due to SVM optimizing hyperplane and complexity. Controversially, compared to previous studies, [1, 4 – 5] presented higher accuracy over 90% of logistic regression and SVM. These studies emphasized that the choice and complexity of feature extraction play an important role in increasing ML’s performance. [1, 5] applied vary features, over 10 features as an input to the ML. Meanwhile, [4] only extracted 5 features with more complexity. This suggests that the complication of feature extraction significantly enhances the efficiency of ML.

# ****Recommendations****

The project studied with a very high dataset, over 200,000 URLs, which significantly increased the time required for ML to run the code and illustrated results. GridSearchCV was used to optimize the ML, further processing time with a default setting of 100 maximum iterations, l1 of penalty, and liblinear of solver. Despite these settings, the accuracy was equivalent to the default configuration. Meanwhile, running GridSearchCV on SVM exceeded 6 hours without results. According to [1, 4 – 5], future work should explore to further enhance the feature extraction by incorporating with more complexity, including JavaScript, greater special characters, HTML Encoding, images etc. This could increase performance to 90%. Moreover, additional machine learning models such as DecisionTree and others should be considered to develop detection performance. With the rise of AI phishing attack, the study of unsupervised ML is highly recommended to address this challenge. 

# ****References****

[1] M. Sameen, K. Han and S. O. Hwang, "PhishHaven—An Efficient Real-Time AI Phishing URLs Detection System," in IEEE Access, vol. 8, pp. 83425-83443, 2020, doi: 10.1109/ACCESS.2020.2991403.
[2] M. Sánchez-Paniagua, E. F. Fernández, E. Alegre, W. Al-Nabki and V. González-Castro, "Phishing URL Detection: A Real-Case Scenario Through Login URLs," in IEEE Access, vol. 10, pp. 42949-42960, 2022, doi: 10.1109/ACCESS.2022.3168681.
[3] MA. Tamal, MK. Islam, T. Bhuiyan, A. Sattar, “Dataset of suspicious phishing URL detection,” in Front. Comput. Sci., vol. 6, doi: 10.3389/fcomp.2024.1308634
[4] V. Vajrobol, B. B. Gupta, A. Gaurav, “Mutual information based logistic regression for phishing URL detection,” in Cyber Security and Applications, vol. 2, pp. 100044, January 1, 2024, doi: 10.1016/j.csa.2024.100044
[5] B. Banik, A. Sarma, “Phishing URL detection system based on URL features using SVM,” in International Journal of Electronics and Applied Research (IJEAR), vol. 5, pp. 40 – 55, December, 2018
