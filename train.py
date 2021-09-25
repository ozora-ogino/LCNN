"""

Train the LCNN and predict.


Note: This is optimized for ASVspoof2019's competition.
      If you wnat to use for your own data  change the database path.

Todo:
    * Select 'feature_type'(fft or cqt).
    * Set the path to 'saving_path' for saving your model.
    * Set the Database path depends on your enviroment.

"""


import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from feature import calc_cqt, calc_stft
from metrics import calculate_eer
from model.lcnn import build_lcnn

# ---------------------------------------------------------------------------------------------------------------------------------------
# model parameters
epochs = 100
batch_size = 256
lr = 0.00001

# We can use 2 types of spectrogram that extracted by using FFT or CQT.
# Set cqt of stft.
feature_type = "cqt"

# The path for saving model
# This is used for ModelChecking callback.
saving_path = "lcnn.h5"
# ---------------------------------------------------------------------------------------------------------------------------------------


# Replace the path to protcol of ASV2019 depending on your environment.
protocol_tr = "./protocol/train_protocol.csv"
protocol_dev = "./protocol/dev_protocol.csv"
protocol_eval = "./protocol/eval_protocol.csv"

# Choose access type PA or LA.
# Replace 'asvspoof_database/ to your database path.
access_type = "PA"
path_to_database = "asvspoof_database/" + access_type
path_tr = path_to_database + "/ASVspoof2019_" + access_type + "_train/flac/"
path_dev = path_to_database + "/ASVspoof2019_" + access_type + "_dev/flac/"
path_eval = path_to_database + "/ASVspoof2019_" + access_type + "_eval/flac/"

if __name__ == "__main__":

    df_tr = pd.read_csv(protocol_tr)
    df_dev = pd.read_csv(protocol_dev)

    if feature_type == "stft":
        print("Extracting train data...")
        x_train, y_train = calc_stft(df_tr, path_tr)
        print("Extracting dev data...")
        x_val, y_val = calc_stft(df_dev, path_dev)

    elif feature_type == "cqt":
        print("Extracting train data...")
        x_train, y_train = calc_cqt(df_tr, path_tr)
        print("Extracting dev data...")
        x_val, y_val = calc_cqt(df_dev, path_dev)

    input_shape = x_train.shape[1:]
    lcnn = build_lcnn(input_shape)

    lcnn.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    cp_cb = ModelCheckpoint(
        filepath="./model",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="auto",
    )

    # Train LCNN
    history = lcnn.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=[x_val, y_val],
        callbacks=[es, cp_cb],
    )
    del x_train, x_val

    print("Extracting eval data")
    df_eval = pd.read_csv(protocol_eval)

    if feature_type == "stft":
        x_eval, y_eval = calc_stft(df_eval, path_eval)

    elif feature_type == "cqt":
        x_eval, y_eval = calc_cqt(df_eval, path_eval)

    # predict
    preds = lcnn.predict(x_eval)

    score = preds[:, 0] - preds[:, 1]  # Get likelihood
    eer = calculate_eer(y_eval, score)  # Get EER score
    print(f"EER : {eer*100} %")
