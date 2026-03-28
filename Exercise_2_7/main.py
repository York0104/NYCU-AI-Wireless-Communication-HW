import argparse
import os

import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

from tools import networks, raputil

K = 64
mu = 2
DEFAULT_SNRS = [5, 10, 15, 20, 25, 30, 35, 40]
DEFAULT_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise 2.7 runner")
    parser.add_argument("--ce-type", choices=["mmse", "dnn"], default="dnn")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="test",
        help="train DNN weights or test CE MSE",
    )
    parser.add_argument("--cp-flag", choices=["true", "false"], default="true")
    parser.add_argument("--training-epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument(
        "--snrs",
        type=int,
        nargs="*",
        default=DEFAULT_SNRS,
        help="SNR values in dB",
    )
    return parser.parse_args()


def run(args):
    ce_type = args.ce_type
    test_ce = args.mode == "test"
    cp_flag = args.cp_flag.lower() == "true"
    snr_list = args.snrs
    training_epochs = args.training_epochs
    batch_size = args.batch_size
    num_trials = args.num_trials

    os.makedirs("dnn_ce", exist_ok=True)

    ber = []
    mse_t_all, mse_f_all = [], []

    for snr in snr_list:
        print("\nSNR=", snr)
        sess, input_holder, output = [], [], []
        if ce_type == "dnn":
            savefile = (
                "dnn_ce/CE_DNN_"
                + ("CPFREE_" if not cp_flag else "")
                + str(2 ** mu)
                + "QAM_SNR_"
                + str(snr)
                + "dB.npz"
            )
            sess, input_holder, output = networks.build_ce_dnn(
                K,
                snr,
                training_epochs=training_epochs,
                batch_size=batch_size,
                savefile=savefile,
                test_flag=test_ce,
                cp_flag=cp_flag,
                nh1=500,
                nh2=250,
            )
        if test_ce:
            mse_t, mse_f = raputil.test_ce(
                sess, input_holder, output, snr, est_type=ce_type, CP_flag=cp_flag, num_trail=num_trials
            )
            mse_t_all.append(mse_t)
            mse_f_all.append(mse_f)
        tf.reset_default_graph()

    print("BER", ber)
    print("MSE_T", mse_t_all)
    print("MSE_F", mse_f_all)

    savefile = "MSE_" + ce_type + "_" + str(2 ** mu) + "QAM" + ("_CP_FREE" if not cp_flag else "")
    if test_ce:
        sio.savemat(savefile + ".mat", {savefile: mse_f_all})


if __name__ == "__main__":
    run(parse_args())
