import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from urllib.parse import urlparse
import urllib.parse
import ipaddress
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy


#checking if URLs use 'http' or 'https'
def has_https(url):
    domain = urlparse(url).netloc
    if 'https' in domain:
        return 1
    else:
        return 0

#count number in the URLs
def digit_num(url):
    i = 0
    for num in url:
        if num.isdigit():
            i += 1
    return i

#Modified by Chat GPT
#Some URLs do not have IP address, fake URLs
def isIP(url):
    domain = urllib.parse.urlparse(url).netloc
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0
    
#from: https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques
def shorten_url(url):
    shorten = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"
    match = re.search(shorten,url)
    if match:
        return 1
    else:
        return 0

#phishing URL contains unique characters
#reference: chat GPT      
def url_entropy(url):
    char_counts = []
    for char in set(url):
        char_counts.append(url.count(char))
    return entropy(char_counts)

#Feature Extraction
#Most phishing URLs consist of:
#wide length,
#a lot of number,
#using http and no IP
#long path (/)
#special characters (modified by chat GPT)
#shorten urls with unusual domain or subdomain
#reference: https://www.geeksforgeeks.org/python-urllib-module/
#different and more than one subdomain
#high entropy: mixing of random characters        
def feature_extraction(content):
    features = {}
    features['URL_Length'] = len(content)
    features['Digits_Num'] = digit_num(content)
    features['Is_HTTP'] = has_https(content)
    features['Has_IP'] = isIP(content)
    features['Domain_Depth'] = content.count('/')
    features['Special_Char'] = len(re.findall(r'[!@#\$%\^&\*\(\),\.\?:\{\}\|<>]',content))
    features['Shorten_Service'] = shorten_url(content)
    features['Subdomain'] = urllib.parse.urlparse(content).netloc.count('.')
    features['URL_Entropy'] = url_entropy(content)
    return features

#extracting URLs from feature extraction
def define_dataframe(data):
    df_features = []
    for url in data['URL']:
        df_features.append(feature_extraction(url))
    df_features = pd.DataFrame(df_features)
    df_features['label'] = data['label']
    return df_features

#Selecting Data
#last column is 0 = phishing and 1 = legitimate 
#normalized data due to its complexity
def Training(data):
    x_data = data.iloc[:,:-1]
    y_data = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    return x_train_scaled, x_test_scaled, y_train, y_test

#reference: https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
def Logistic_Regression(x_train,x_test,y_train,y_test):
    model = LogisticRegression()
    model.fit(x_train,y_train)
    print('----Logistic Regression Model----')
    y_pred, accuracy, precision, recall, f1 = evaluation(model, x_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Phishing', "Legitimate"], yticklabels=["Phishing", "Legitimate"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('LogisticRegression')
    plt.show()

def SVM_model(x_train,x_test,y_train,y_test):
    model = SVC()
    model.fit(x_train,y_train)
    print('----Support Vector Machine Model----')
    y_pred, accuracy, precision, recall, f1 = evaluation(model, x_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Phishing', "Legitimate"], yticklabels=["Phishing", "Legitimate"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Support Vector Machine')
    plt.show()

#measure performance
#true positive is mainly considered
#confusion matrix provides TP, TN, FP, FN --> can compute the rate
def evaluation(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    TN_data, FP_data, FN_data, TP_data = cm.ravel()
    TP = TP_data / (TP_data + FN_data)
    TN = TN_data / (TN_data + FP_data)
    FP = FP_data / (TN_data + FP_data)
    FN = FN_data / (TP_data + FN_data)
    print(f'Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1_score: {f1}')
    print(f'TP: {TP}\n TN: {TN}\n FP: {FP}\n FN: {FN}')
    return y_pred, accuracy, precision, recall, f1

#Step 1: Read file
#Step 2: Selecting data and extracting features
#Step 3: Training Data
#Step 4: ML models
#Step 5: Evaluation
if __name__ == '__main__':
    data = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
    df = define_dataframe(data)
    x_train, x_test, y_train, y_test = Training(df)
    Logistic_Regression(x_train, x_test, y_train, y_test )
    SVM_model(x_train, x_test, y_train, y_test )
    
