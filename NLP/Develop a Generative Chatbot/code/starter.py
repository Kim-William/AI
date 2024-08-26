import logging
import pandas as pd
from modelling import model
from attnModelling import attnModel
from predict import predict
# ref: https://colab.research.google.com/drive/1I60_OAeamcRE7VWb7JLCDhlAO4vZBahl#scrollTo=qqttxnJ2_qBS

data_size = 1500

train_df = pd.read_csv("./preprocessed/train_cleaned.csv").head(data_size)
test_df = pd.read_csv("./preprocessed/test_cleaned.csv").head(data_size)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Data is loaded:")
logging.info(train_df.shape)

# model(train_df, test_df)
# attnModel(train_df, test_df)
predict(data_size)