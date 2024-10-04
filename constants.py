import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

sizOfTest = 0.2
stateOfRandomness = 42

SHAPEOFINPUT = (12,)
UNITSLAYER1 = 64
UNITSLAYER2 = 32
RATEOFDROPOUT = 0.5
OUTPUTUNITS = 3 
HIDDENACTIVATION = 'relu'
OUTPUTACTIVATION = 'softmax'
EPOCHSAMOUNT = 20
BATCHSIZE = 32
RATESOFLEARNING = [1e-2, 1e-3, 1e-4]
UNITSLAYEROPTIONS1 = [64, 128, 256]
UNITSLAYEROPTIONS2 = [128]
DROPOUTRATES = [0.5]
OPTIMIZERS = ['Rprop', 'Adam', 'SGD']


FILE = 'Almond.csv'
CSV_FILE = pd.read_csv(FILE)

if 'Unnamed: 0' in CSV_FILE.columns:
    CSV_FILE = CSV_FILE.drop(columns=['Unnamed: 0'])
    
X = CSV_FILE.drop(columns=['Type'])
y = CSV_FILE['Type']

IMPUTER = KNNImputer(n_neighbors=5)
XIMPUTED = pd.DataFrame(IMPUTER.fit_transform(X), columns=X.columns)

ENCODER = OneHotEncoder()
YENCODEDSPARSE = ENCODER.fit_transform(y.values.reshape(-1, 1))
YENCODED = YENCODEDSPARSE.toarray()

SCALER = StandardScaler()
XSCALED = SCALER.fit_transform(XIMPUTED)

TRAINX, TESTX, TRAINY, TESTY = train_test_split(XSCALED, YENCODED, test_size=sizOfTest, random_state=stateOfRandomness)