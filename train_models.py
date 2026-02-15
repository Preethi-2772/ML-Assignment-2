{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "baa4d09f-1161-4145-af57-35d157914172",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in /home/cloud/anaconda3/lib/python3.12/site-packages (2.2.3)\n",
      "Requirement already satisfied: numpy in /home/cloud/anaconda3/lib/python3.12/site-packages (1.26.4)\n",
      "Requirement already satisfied: scikit-learn in /home/cloud/anaconda3/lib/python3.12/site-packages (1.7.2)\n",
      "Requirement already satisfied: matplotlib in /home/cloud/anaconda3/lib/python3.12/site-packages (3.10.1)\n",
      "Requirement already satisfied: seaborn in /home/cloud/anaconda3/lib/python3.12/site-packages (0.13.2)\n",
      "Requirement already satisfied: xgboost in /home/cloud/anaconda3/lib/python3.12/site-packages (3.2.0)\n",
      "Requirement already satisfied: streamlit in /home/cloud/anaconda3/lib/python3.12/site-packages (1.54.0)\n",
      "Requirement already satisfied: joblib in /home/cloud/anaconda3/lib/python3.12/site-packages (1.4.2)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /home/cloud/anaconda3/lib/python3.12/site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in /home/cloud/anaconda3/lib/python3.12/site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: scipy>=1.8.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from scikit-learn) (1.16.3)\n",
      "Requirement already satisfied: threadpoolctl>=3.1.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from scikit-learn) (3.6.0)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.3.1)\n",
      "Requirement already satisfied: cycler>=0.10 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (0.12.1)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (4.56.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (1.4.8)\n",
      "Requirement already satisfied: packaging>=20.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (24.1)\n",
      "Requirement already satisfied: pillow>=8 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (11.0.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from matplotlib) (3.2.1)\n",
      "Requirement already satisfied: nvidia-nccl-cu12 in /home/cloud/anaconda3/lib/python3.12/site-packages (from xgboost) (2.29.3)\n",
      "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<7,>=4.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (6.0.0)\n",
      "Requirement already satisfied: blinker<2,>=1.5.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (1.9.0)\n",
      "Requirement already satisfied: cachetools<7,>=5.5 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (6.2.6)\n",
      "Requirement already satisfied: click<9,>=7.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (3.1.46)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (0.9.1)\n",
      "Requirement already satisfied: protobuf<7,>=3.20 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (4.25.3)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (23.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: tenacity<10,>=8.1.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (9.1.4)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (6.4.1)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.10.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (4.15.0)\n",
      "Requirement already satisfied: watchdog<7,>=2.1.5 in /home/cloud/anaconda3/lib/python3.12/site-packages (from streamlit) (4.0.1)\n",
      "Requirement already satisfied: jinja2 in /home/cloud/anaconda3/lib/python3.12/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: narwhals>=1.27.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.16.0)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
      "Requirement already satisfied: six>=1.5 in /home/cloud/anaconda3/lib/python3.12/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /home/cloud/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /home/cloud/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2.2.3)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /home/cloud/anaconda3/lib/python3.12/site-packages (from requests<3,>=2.27->streamlit) (2024.12.14)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from jinja2->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in /home/cloud/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (23.1.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /home/cloud/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in /home/cloud/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in /home/cloud/anaconda3/lib/python3.12/site-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<7,>=4.0->streamlit) (0.10.6)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pandas numpy scikit-learn matplotlib seaborn xgboost streamlit joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b01765a3-9329-44d5-a14e-e1c1d7787b9f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    precision_score,\n",
    "    recall_score,\n",
    "    f1_score,\n",
    "    roc_auc_score,\n",
    "    matthews_corrcoef,\n",
    "    confusion_matrix\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "06c2c6d9-8c65-4137-9ba3-691d1842bddb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(1025, 14)\n",
      "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
      "0   52    1   0       125   212    0        1      168      0      1.0      2   \n",
      "1   53    1   0       140   203    1        0      155      1      3.1      0   \n",
      "2   70    1   0       145   174    0        1      125      1      2.6      0   \n",
      "3   61    1   0       148   203    0        1      161      0      0.0      2   \n",
      "4   62    0   0       138   294    1        1      106      0      1.9      1   \n",
      "\n",
      "   ca  thal  target  \n",
      "0   2     3       0  \n",
      "1   0     3       0  \n",
      "2   0     3       0  \n",
      "3   1     3       0  \n",
      "4   3     2       0  \n"
     ]
    }
   ],
   "source": [
    "# Loading Dataset\n",
    "df = pd.read_csv(\"Assignment2_data/heart.csv\")\n",
    "\n",
    "print(df.shape)\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "89502d70-4123-4695-becb-34e0a174679a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature / Target Split\n",
    "X = df.drop(\"target\", axis=1)\n",
    "y = df[\"target\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "51f949e9-a218-48d6-b52e-7f8a9e2f9789",
   "metadata": {},
   "outputs": [],
   "source": [
    "# train test split \n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5f0f02ca-036b-4fcc-bb5f-910f9dede488",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['model/scaler.pkl']"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler = StandardScaler()\n",
    "\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "joblib.dump(scaler, \"model/scaler.pkl\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2e7244bc-ff3b-4241-92d5-326624c20af7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# logistic regression \n",
    "lr = LogisticRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "pred_lr = lr.predict(X_test)\n",
    "prob_lr = lr.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "5c917ef4-7c96-4338-97e1-e659c1ce7ff0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# decision tree\n",
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train, y_train)\n",
    "pred_dt = dt.predict(X_test)\n",
    "prob_dt = dt.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "516d733c-d38f-4dc7-88bf-5f6f59e2f4d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# knn\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)\n",
    "pred_knn = knn.predict(X_test)\n",
    "prob_knn = knn.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "71c0a579-77c5-4afa-a014-69ff7786ef7c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Naive Bayes\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "pred_nb = nb.predict(X_test)\n",
    "prob_nb = nb.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d99f0cdd-12d5-47ef-a112-dad09ee554b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# random forest\n",
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "rf.fit(X_train, y_train)\n",
    "pred_rf = rf.predict(X_test)\n",
    "prob_rf = rf.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "a8672173-5c17-4d5c-9586-36ccd55a97c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# xgboost\n",
    "xgb = XGBClassifier(eval_metric='logloss')\n",
    "xgb.fit(X_train, y_train)\n",
    "pred_xgb = xgb.predict(X_test)\n",
    "prob_xgb = xgb.predict_proba(X_test)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "bd02e044-fba2-46c1-b42b-2bf62dcb891e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate(y_test, pred, prob):\n",
    "    return [\n",
    "        accuracy_score(y_test, pred),\n",
    "        roc_auc_score(y_test, prob),\n",
    "        precision_score(y_test, pred),\n",
    "        recall_score(y_test, pred),\n",
    "        f1_score(y_test, pred),\n",
    "        matthews_corrcoef(y_test, pred)\n",
    "    ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c57134e4-469f-4c88-a2cf-1a5379769adb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                 Model  Accuracy       AUC  Precision    Recall        F1  \\\n",
      "0  Logistic Regression  0.795122  0.878736   0.756303  0.873786  0.810811   \n",
      "1        Decision Tree  0.985366  0.985437   1.000000  0.970874  0.985222   \n",
      "2                  KNN  0.834146  0.948553   0.800000  0.893204  0.844037   \n",
      "3          Naive Bayes  0.800000  0.870550   0.754098  0.893204  0.817778   \n",
      "4        Random Forest  0.985366  1.000000   1.000000  0.970874  0.985222   \n",
      "5              XGBoost  0.985366  0.989435   1.000000  0.970874  0.985222   \n",
      "\n",
      "        MCC  \n",
      "0  0.597255  \n",
      "1  0.971151  \n",
      "2  0.672727  \n",
      "3  0.610224  \n",
      "4  0.971151  \n",
      "5  0.971151  \n"
     ]
    }
   ],
   "source": [
    "results = pd.DataFrame(columns=[\n",
    "    \"Model\",\"Accuracy\",\"AUC\",\"Precision\",\"Recall\",\"F1\",\"MCC\"\n",
    "])\n",
    "\n",
    "results.loc[len(results)] = [\"Logistic Regression\", *evaluate(y_test,pred_lr,prob_lr)]\n",
    "results.loc[len(results)] = [\"Decision Tree\", *evaluate(y_test,pred_dt,prob_dt)]\n",
    "results.loc[len(results)] = [\"KNN\", *evaluate(y_test,pred_knn,prob_knn)]\n",
    "results.loc[len(results)] = [\"Naive Bayes\", *evaluate(y_test,pred_nb,prob_nb)]\n",
    "results.loc[len(results)] = [\"Random Forest\", *evaluate(y_test,pred_rf,prob_rf)]\n",
    "results.loc[len(results)] = [\"XGBoost\", *evaluate(y_test,pred_xgb,prob_xgb)]\n",
    "\n",
    "print(results)\n",
    "results.to_csv(\"model_results.csv\", index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "4876782d-74fd-4a3d-b881-c110fa6ec3cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['model/xgb.pkl']"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(lr, \"model/logistic.pkl\")\n",
    "joblib.dump(dt, \"model/dt.pkl\")\n",
    "joblib.dump(knn, \"model/knn.pkl\")\n",
    "joblib.dump(nb, \"model/nb.pkl\")\n",
    "joblib.dump(rf, \"model/rf.pkl\")\n",
    "joblib.dump(xgb, \"model/xgb.pkl\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
