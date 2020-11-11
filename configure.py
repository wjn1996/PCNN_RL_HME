import argparse
import sys
import os


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--data_path", default="./data/",
                        type=str, help="Path of train data")
    parser.add_argument("--model_dict", default="./model/",
                        type=str, help="Path of train data")
    parser.add_argument("--max_len", default=70,
                        type=int, help="Max sentence length in data (unchangeable)")

    # Model Hyper-parameters
    # Embeddings
    parser.add_argument("--word_dim", default=50,
                        type=int, help="Dimensionality of word embedding (default: 50, unchangeable)")
    parser.add_argument("--pos_dim", default=5,
                        type=int, help="Dimensionality of relative position embedding (default: 5. unchangeable)")
    parser.add_argument("--rel_dim", default=50,
                        type=int, help="relation label embedding size (default: 20)")
    parser.add_argument("--emb_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of embedding layer (default: 0.7)")
    parser.add_argument("--sent_dim", default=100,
                        type=int, help="Dimensionality of sentence hidden (default: 100)")
    parser.add_argument("--ent_dim", default=50,
                        type=int, help="Dimensionality of entity embedding size (default: 200)")
    parser.add_argument("--hie_dim", default=50,
                        type=int, help="Dimensionality of hierarchical memory cell embedding size (default: 200)")

    parser.add_argument("--gamma", default=0.25,
                        type=float, help="the margin threshold of hierarchical similarity range from (0.25,0.5,0.75)")

    # Misc
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--drop_out", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=40,
                        type=int, help="Batch Size (default: 160)")
    parser.add_argument("--num_epochs", default=10,
                        type=int, help="Number of training epochs (Default: 50)")
    parser.add_argument("--num_pretrain_epochs", default=10,
                        type=int, help="Number of pretraining epochs (Default: 5)")
    parser.add_argument("--num_mc_sample", default=3,
                        type=int, help="Number of monte carlo sample (Default: 3)")
    parser.add_argument("--display_every", default=10,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=1169,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--policy_learning_rate", default=1.0,
                        type=float, help="The learning rate of policy (Default: 1.0)")
    parser.add_argument("--extractor_learning_rate", default=1.0,
                        type=float, help="The learning rate of extractor (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.9,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")
    parser.add_argument("--epsilon", default=0.9,
                        type=float, help="the epsilon of select opim action (Default: 0.9)")
    
    

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--use_gpu", default=False,
                        type=bool, help="Allow gpu memory growth")
    parser.add_argument("--use_pcnn", default=False,
                        type=bool, help="Allow PCNN")
    parser.add_argument("--test_select", default=False,
                        type=bool, help="select or un-select the credible test data before valuate")

    # Visualization Parameters
    parser.add_argument("--checkpoint_dir", default=None,
                        type=str, help="Visualize this checkpoint") 
    # ranking loss
    parser.add_argument("--lm", default=1.0,
                        type=float, help="lambda")
    parser.add_argument("--margin_plus", default=2.5,
                        type=float, help="")
    parser.add_argument("--margin_minus", default=0.5,
                        type=float, help="")
    # mould select
    parser.add_argument("--filters_num", default=230,
                        type=int, help="CNN filter number")
    parser.add_argument("--filters", default=[3],
                        type=list, help="CNN filters")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args

args = parse_args()


if not os.path.exists(args.model_dict):
    os.makedirs(args.model_dict)
if not os.path.exists('./data'):
    os.makedirs('./data')
if not os.path.exists('./corpus'):
    os.makedirs('./corpus')