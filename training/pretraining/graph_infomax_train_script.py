from functools import partial
from models.pretraining.encoders import RevGATConvEncoder
from models.pretraining.graph_infomax import random_sample_corruption, readout_function,  DeepGraphInfomaxWrapper as DGI

# TODO get the training set properly
TRAINING_SET = None

CORRUPTION_FUNC = partial(
    random_sample_corruption,
    TRAINING_SET
)

REV_GAT_ENCODER = RevGATConvEncoder(
    in_channels      = 0,
    hidden_channels  = 0,
    out_channels     = 0,
    num_convs        = 1,
    dropout          = 0.0,
    project          = False,
    root_weight      = True,
    aggr             = "mean",
    num_groups       = 2,
    normalize_hidden = True
)


def main():
    rev_gat_dgi = DGI(
        in_channels      = 0,
        hidden_channels  = 0,
        out_channels     = 0,
        encoder          = REV_GAT_ENCODER,
        normalize_hidden = True,
        readout          = readout_function,
        corruption       = CORRUPTION_FUNC,
        dropout          = 0.0
    )


if __name__ == "__main__":
    main()



