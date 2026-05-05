from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

columns = [
    'duration','protocol_type','service','flag','src_bytes',
    'dst_bytes','land','wrong_fragment','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell',
    'su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

df_train = pd.read_csv('KDDTrain+.txt', names=columns)
df_test  = pd.read_csv('KDDTest+.txt',  names=columns)

df_train['target'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['target']  = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col]  = le.fit_transform(df_test[col])

X_train = df_train.drop(['label', 'difficulty', 'target'], axis=1)
y_train = df_train['target']

X_test = df_test.drop(['label', 'difficulty', 'target'], axis=1)
y_test = df_test['target']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)

X_normal = X_train_scaled[y_train == 0]
model_if = IsolationForest(contamination=0.4, random_state=42)
model_if.fit(X_normal)

y_hybrid = model_rf.predict(X_test_scaled)

probas = model_rf.predict_proba(X_test_scaled)
normal_confidence = probas[:, 0]

uncertain_normal = (y_hybrid == 0) & (normal_confidence < 0.95)

if_predictions = model_if.predict(X_test_scaled[uncertain_normal])
if_predictions_converted = np.where(if_predictions == 1, 0, 1)

y_hybrid[uncertain_normal] = if_predictions_converted

print(classification_report(y_test, y_hybrid,
      target_names=['Normal', 'Suspect'],
      digits=4))

cm = confusion_matrix(y_test, y_hybrid)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Normal', 'Suspect'],
            yticklabels=['Normal', 'Suspect'])
plt.title('Confusion Matrix')
plt.ylabel('Real')
plt.xlabel('Predicted')
plt.show()