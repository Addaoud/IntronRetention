import torch.nn as nn
from typing import Optional
import torch
import numpy as np
from scipy.interpolate import splev
from torch import einsum
from transformers import AutoModel
from src.seed import set_seed

set_seed()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        output = self.linear(x)
        return output


def bs(x, df=None, knots=None, degree=3, intercept=False):
    """
    df : int
        The number of degrees of freedom to use for this spline. The
        return value will have this many columns. You must specify at least
        one of `df` and `knots`.
    knots : list(float)
        The interior knots of the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of `df` and `knots`.
    degree : int
        The degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, the resulting spline basis will span the intercept term
        (i.e. the constant function). If `False` (the default) then this
        will not be the case, which is useful for avoiding overspecification
        in models that include multiple spline terms and/or an intercept term.
    """
    order = degree + 1
    inner_knots = []
    if df is not None and knots is None:
        n_inner_knots = df - order + (1 - intercept)
        if n_inner_knots < 0:
            n_inner_knots = 0
            print("df was too small; have used %d" % (order - (1 - intercept)))
        if n_inner_knots > 0:
            inner_knots = np.percentile(
                x, 100 * np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            )
    elif knots is not None:
        inner_knots = knots
    all_knots = np.concatenate(([np.min(x), np.max(x)] * order, inner_knots))
    all_knots.sort()
    n_basis = len(all_knots) - (degree + 1)
    basis = np.empty((x.shape[0], n_basis), dtype=float)
    for i in range(n_basis):
        coefs = np.zeros((n_basis,))
        coefs[i] = 1
        basis[:, i] = splev(x, (all_knots, coefs, degree))
    if not intercept:
        basis = basis[:, 1:]
    return basis


def spline_factory(n, df, log=False):
    if log:
        dist = np.array(np.arange(n) - n / 2.0)
        dist = np.log(np.abs(dist) + 1) * (2 * (dist > 0) - 1)
        n_knots = df - 4
        knots = np.linspace(np.min(dist), np.max(dist), n_knots + 2)[1:-1]
        return torch.from_numpy(bs(dist, knots=knots, intercept=True)).float()
    else:
        dist = np.arange(n)
        return torch.from_numpy(bs(dist, df=df, intercept=True)).float()


class BSplineTransformation(nn.Module):
    def __init__(self, degrees_of_freedom, log=False, scaled=False):
        super(BSplineTransformation, self).__init__()
        self._spline_tr = None
        self._log = log
        self._scaled = scaled
        self._df = degrees_of_freedom

    def forward(self, input):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        return torch.matmul(input, self._spline_tr)


class Sei(nn.Module):
    def __init__(self, sequence_length=600, n_genomic_features=21907):
        """
        Parameters
        ----------
        sequence_length : int
        n_genomic_features : int
        """
        super(Sei, self).__init__()
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True),
        )
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True),
        )
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True),
        )
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True),
        )
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True),
        )
        self._spline_df = int(128 / 8)
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5), BSplineTransformation(self._spline_df, scaled=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, n_genomic_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_genomic_features, n_genomic_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward propagation of a batch.
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        predict = self.classifier(reshape_out)
        return predict


def generate_SEI():
    net = Sei()
    model_pretrained_dict = torch.load("sei.pth")
    keys_pretrained = list(model_pretrained_dict.keys())
    keys_net = list(net.state_dict())
    model_weights = net.state_dict()
    for i in range(len(keys_net)):
        model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
    net.load_state_dict(model_weights)
    print("Model succesfully loaded with pretrained weights")
    return net


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))  # 960*960

    def forward(self, x):
        attn_logits = einsum(
            "b n d, d e -> b n e", x, self.to_attn_logits
        )  # 64*16*960 , 960*960
        attn = attn_logits.softmax(dim=-2)  # 64*1*16*960 => 64*1*16*960
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class finetuneblock(nn.Module):
    def __init__(
        self, hidden_dim: int, embed_dim: int, kernel_size: int, output_dim: int
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=kernel_size),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.attention_pool = AttentionPool(embed_dim)
        self.fcn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.1)
        )
        self.prediction_head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.project(x)
        x = x.permute(0, 2, 1)
        x = self.attention_pool(x)
        x = self.fcn(x)
        x = self.prediction_head(x)
        return x


class FSei(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        kernel_size: int,
        n_genomic_features: Optional[int] = 2,
        FCNN: Optional[int] = 160,
    ):
        """
        Parameters
        ----------
        FCNN : int
        hidden_dim: int
        embed_dim: int
        kernel_size: int
        n_genomic_features : int
        """
        super(FSei, self).__init__()
        self.FCNN = FCNN
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.n_genomic_features = n_genomic_features
        self.max = nn.MaxPool1d(kernel_size=4, stride=4)
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 3 * self.FCNN, kernel_size=9, padding=4),
            nn.Conv1d(3 * self.FCNN, 3 * self.FCNN, kernel_size=9, padding=4),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(3 * self.FCNN, 3 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(3 * self.FCNN, 3 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(3 * self.FCNN, 4 * self.FCNN, kernel_size=9, padding=4),
            nn.Conv1d(4 * self.FCNN, 4 * self.FCNN, kernel_size=9, padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(4 * self.FCNN, 4 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(4 * self.FCNN, 4 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.lconv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(4 * self.FCNN, 6 * self.FCNN, kernel_size=9, padding=4),
            nn.Conv1d(6 * self.FCNN, 6 * self.FCNN, kernel_size=9, padding=4),
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(6 * self.FCNN, 6 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(6 * self.FCNN, 6 * self.FCNN, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
        )
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(
                6 * self.FCNN, 6 * self.FCNN, kernel_size=5, dilation=2, padding=4
            ),
            nn.ReLU(inplace=True),
        )
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(
                6 * self.FCNN, 6 * self.FCNN, kernel_size=5, dilation=4, padding=8
            ),
            nn.ReLU(inplace=True),
        )
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(
                6 * self.FCNN, 6 * self.FCNN, kernel_size=5, dilation=8, padding=16
            ),
            nn.ReLU(inplace=True),
        )
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(
                6 * self.FCNN, 6 * self.FCNN, kernel_size=5, dilation=16, padding=32
            ),
            nn.ReLU(inplace=True),
        )
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(
                6 * self.FCNN, 6 * self.FCNN, kernel_size=5, dilation=25, padding=50
            ),
            nn.ReLU(inplace=True),
        )
        self._spline_df = 16

        self.spline_tr = nn.Sequential(
            BSplineTransformation(self._spline_df, scaled=False)
        )
        self.classifier = finetuneblock(
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
            kernel_size=self.kernel_size,
            output_dim=self.n_genomic_features,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward propagation of a batch.
        """
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)
        lout2 = self.lconv2(self.max(out1 + lout1))
        out2 = self.conv2(lout2)
        lout3 = self.lconv3(self.max(out2 + lout2))
        out3 = self.conv3(lout3)
        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5
        spline_out = self.spline_tr(out)
        output = self.classifier(spline_out)
        return output


def generate_FSei(
    new_model: bool,
    use_pretrain: Optional[bool] = False,
    freeze_weights: Optional[bool] = False,
    model_path: Optional[str] = None,
):
    hidden_dim = 960
    embed_dim = 520
    kernel_size = 3
    n_genomic_features = 2
    FCNN = 160
    if new_model:
        net = FSei(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            n_genomic_features=n_genomic_features,
            FCNN=FCNN,
        )
    else:
        net = torch.load(model_path)
    if use_pretrain:
        model_pretrained_dict = torch.load("sei.pth")
        keys_pretrained = list(model_pretrained_dict.keys())[:34]
        keys_net = list(net.state_dict())[:34]
        model_weights = net.state_dict()
        for i in range(len(keys_net)):
            model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
        net.load_state_dict(model_weights)
        if freeze_weights:
            model_params = list(net.parameters())
            for i in range(len(keys_net)):
                model_params[i].requires_grad = False
        print("Model succesfully loaded with pretrained weights")
    return net


class FDNABert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        kernel_size: int,
        n_genomic_features: int,
    ):
        super(FDNABert, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.n_genomic_features = n_genomic_features
        self.pretrained_model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        self.pretrained_model.pooler = None
        self.classifier = finetuneblock(
            hidden_dim=self.hidden_dim,
            embed_dim=self.embed_dim,
            kernel_size=self.kernel_size,
            output_dim=self.n_genomic_features,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        embeddings = self.pretrained_model(input_ids, attention_mask=attention_mask)[0]
        output = self.classifier(embeddings)
        return output


def generate_FDNABert(
    freeze_weights: bool,
    model_path: Optional[str] = None,
):
    hidden_dim = 150
    embed_dim = 520
    kernel_size = 16
    net = FDNABert(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        n_genomic_features=2,
    )
    if freeze_weights:
        model_params = list(net.parameters())
        for i in range(135):
            model_params[i].requires_grad = False
    print("Model succesfully built")
    return net


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class Basset(nn.Module):
    def __init__(self, params, wvmodel=None, useEmbeddings=False):
        super(Basset, self).__init__()
        self.CNN1filters = params["CNN1_filters"]
        self.CNN1filterSize = params["CNN1_filtersize"]
        self.CNN1poolSize = params["CNN1_poolsize"]
        self.CNN1padding = params["CNN1_padding"]
        self.CNN1useExponential = params["CNN1_useexponential"]
        self.CNN2filters = params["CNN2_filters"]
        self.CNN2filterSize = params["CNN2_filtersize"]
        self.CNN2poolSize = params["CNN2_poolsize"]
        self.CNN2padding = params["CNN2_padding"]
        self.CNN3filters = params["CNN3_filters"]
        self.CNN3filterSize = params["CNN3_filtersize"]
        self.CNN3poolSize = params["CNN3_poolsize"]
        self.CNN3padding = params["CNN3_padding"]
        self.FC1inputSize = params["FC1_inputsize"]
        self.FC1outputSize = params["FC1_outputsize"]
        self.FC2outputSize = params["FC2_outputsize"]
        self.numClasses = params["num_classes"]

        self.useEmbeddings = useEmbeddings
        if not self.useEmbeddings:
            self.numInputChannels = params[
                "input_channels"
            ]  # number of channels, one hot encoding
        else:
            self.embSize = params["embd_size"]
            weights = torch.FloatTensor(wvmodel.wv.vectors)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            self.numInputChannels = self.embSize

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.numInputChannels,
                out_channels=self.CNN1filters,
                kernel_size=self.CNN1filterSize,
                padding=self.CNN1padding,
                bias=False,
            ),  # if using batchnorm, no need to use bias in a CNN
            nn.BatchNorm1d(num_features=self.CNN1filters),
            nn.ReLU() if self.CNN1useExponential == False else Exponential(),
            nn.MaxPool1d(kernel_size=self.CNN1poolSize),
        )
        self.dropout1 = nn.Dropout(p=0.2)

        self.layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.CNN1filters,
                out_channels=self.CNN2filters,
                kernel_size=self.CNN2filterSize,
                padding=self.CNN2padding,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=self.CNN2filters),
            nn.ReLU(),
        )
        self.dropout2 = nn.Dropout(p=0.2)

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.CNN2filters,
                out_channels=self.CNN3filters,
                kernel_size=self.CNN3filterSize,
                padding=self.CNN3padding,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=self.CNN3filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.CNN3poolSize),
        )
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(
            in_features=self.FC1inputSize, out_features=self.FC1outputSize
        )
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(
            in_features=self.FC1outputSize, out_features=self.FC2outputSize
        )
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(
            in_features=self.FC2outputSize, out_features=self.numClasses
        )

    def forward(self, inputs):
        if self.useEmbeddings:
            output = self.embedding(inputs)
            output = output.permute(0, 2, 1)
        else:
            output = inputs
        output = self.layer1(output)
        output = self.dropout1(output)
        output = self.layer2(output)
        output = self.dropout2(output)
        output = self.layer3(output)
        output = self.dropout3(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = self.relu4(output)
        output = self.dropout4(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.dropout5(output)
        output = self.fc3(output)
        assert not torch.isnan(output).any()
        return output
