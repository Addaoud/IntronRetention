import torch.nn as nn
from typing import Optional, Dict
import torch
import numpy as np
from scipy.interpolate import splev
from torch import einsum
from src.seed import set_seed
from transformers import BertModel, BertForSequenceClassification

set_seed()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input: torch.Tensor):
        output = self.linear(input)
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

    def forward(self, input: torch.Tensor):
        if self._spline_tr is None:
            spatial_dim = input.size()[-1]
            self._spline_tr = spline_factory(spatial_dim, self._df, log=self._log)
            if self._scaled:
                self._spline_tr = self._spline_tr / spatial_dim
            if input.is_cuda:
                self._spline_tr = self._spline_tr.cuda()
        return torch.matmul(input, self._spline_tr)


class Sei(nn.Module):
    def __init__(self, n_genomic_features=21907):
        """
        Parameters
        ----------
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

    def forward(self, input: torch.Tensor):
        """
        Forward propagation of a batch.
        """
        lout1 = self.lconv1(input)
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


def build_SEI():
    net = Sei()
    model_pretrained_dict = torch.load("sei.pth")
    keys_pretrained = list(model_pretrained_dict.keys())
    keys_net = list(net.state_dict())
    model_weights = net.state_dict()
    for i in range(len(keys_net)):
        model_weights[keys_net[i]] = model_pretrained_dict[keys_pretrained[i]]
    net.load_state_dict(model_weights)
    print("Model loaded with pretrained weights")
    return net


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_attn_logits = nn.Parameter(torch.eye(dim))  # 960*960

    def forward(self, x: torch.Tensor):
        attn_logits = einsum(
            "b n d, d e -> b n e", x, self.to_attn_logits
        )  # 64*16*960 , 960*960
        attn = attn_logits.softmax(dim=-2)  # 64*1*16*960 => 64*1*16*960
        return (x * attn).sum(dim=-2).squeeze(dim=-2)


class finetuneblock1(nn.Module):
    def __init__(
        self,
        input_channels: int,
        kernel_size: int,
        embed_dim: int,
        feedforward_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(input_channels, embed_dim, kernel_size=kernel_size),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.attention_pool = AttentionPool(embed_dim)
        # self.max = nn.MaxPool1d(kernel_size=)
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(feedforward_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.prediction_head = nn.Linear(embed_dim, output_dim)

    def forward(self, input):
        x = self.project(input)
        x = x.permute(0, 2, 1)
        x = self.attention_pool(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.prediction_head(x)
        return x


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
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.2)
        )
        self.prediction_head = nn.Linear(embed_dim, output_dim)

    def forward(self, input):
        x = self.project(input)
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
        """self.classifier = nn.Sequential(
            nn.Linear(960 * self._spline_df, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, n_genomic_features),
        )"""

    def forward(self, input: torch.Tensor):
        """
        Forward propagation of a batch.
        """
        lout1 = self.lconv1(input)
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
        # reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)
        # output = self.classifier(reshape_out)
        return output


def build_FSei(
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
    net = FSei(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        n_genomic_features=n_genomic_features,
        FCNN=FCNN,
    )
    if not new_model and model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if use_pretrain and new_model:
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
        print("Model loaded with pretrained weights")
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
        self.pretrained_model = BertModel.from_pretrained(
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
        input_ids: torch.Tensor = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        embeddings = self.pretrained_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )[0].transpose(1, 2)
        output = self.classifier(embeddings)
        return output


def build_FDNABert(
    new_model: bool,
    freeze_weights: bool,
    model_path: Optional[str] = None,
):
    hidden_dim = 768
    embed_dim = 960
    kernel_size = 5
    net = FDNABert(
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        kernel_size=kernel_size,
        n_genomic_features=2,
    )
    if not new_model and model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if freeze_weights:
        model_params = list(net.parameters())
        for i in range(135):
            model_params[i].requires_grad = False
        print("Freezing model's pre-trained weights")
    print("Model succesfully built")
    return net


class DNABERT2(nn.Module):
    def __init__(self, n_genomic_features) -> None:
        super(DNABERT2, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "zhihan1996/DNABERT-2-117M",
            cache_dir=None,
            num_labels=n_genomic_features,
            trust_remote_code=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits
        return output


def build_DNABert(
    new_model: bool,
    freeze_weights: bool,
    model_path: Optional[str] = None,
    n_genomic_features: Optional[int] = 2,
):
    net = DNABERT2(n_genomic_features=n_genomic_features)
    if not new_model and model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if freeze_weights:
        model_params = list(net.parameters())
        for i in range(135):
            model_params[i].requires_grad = False
        print("Freezing model's pre-trained weights")
    print("Model succesfully built")
    return net


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class Basset(nn.Module):
    def __init__(self, model: Dict[str, int]):
        super(Basset, self).__init__()
        self.CNN1filters = model["CNN1filters"]
        self.CNN1filterSize = model["CNN1filterSize"]
        self.CNN1poolSize = model["CNN1poolSize"]
        self.CNN1padding = model["CNN1padding"]
        self.CNN2filters = model["CNN2filters"]
        self.CNN2filterSize = model["CNN2filterSize"]
        self.CNN2poolSize = model["CNN2poolSize"]
        self.CNN2padding = model["CNN2padding"]
        self.CNN3filters = model["CNN3filters"]
        self.CNN3filterSize = model["CNN3filterSize"]
        self.CNN3poolSize = model["CNN3poolSize"]
        self.CNN3padding = model["CNN3padding"]
        self.FC1inputSize = model["FC1inputSize"]
        self.FC1outputSize = model["FC1outputSize"]
        self.FC2outputSize = model["FC2outputSize"]
        self.numClasses = model["numClasses"]

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=self.CNN1filters,
                kernel_size=self.CNN1filterSize,
                padding=self.CNN1padding,
                bias=False,
            ),  # if using batchnorm, no need to use bias in a CNN
            nn.BatchNorm1d(num_features=self.CNN1filters),
            nn.ReLU(),
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

    def forward(self, input):
        output = self.layer1(input)
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


class ConvNet(nn.Module):
    def __init__(self, n_features: int):
        super(ConvNet, self).__init__()
        self.n_features = n_features
        self.lconv_network = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=11, padding=5),
            nn.Conv1d(480, 480, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=16, stride=8),
            nn.Conv1d(480, 640, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(640),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Conv1d(640, 720, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(720),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Conv1d(720, 720, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Conv1d(720, 960, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(960),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.dconv1 = nn.Sequential(
            nn.Conv1d(960, 1280, kernel_size=5, dilation=2, padding=4, bias=False),
            nn.BatchNorm1d(1280),
            nn.GELU(),
            nn.Conv1d(1280, 960, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )
        self.dconv2 = nn.Sequential(
            nn.Conv1d(960, 1280, kernel_size=5, dilation=4, padding=8, bias=False),
            nn.BatchNorm1d(1280),
            nn.GELU(),
            nn.Conv1d(1280, 960, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv1d(960, 1280, kernel_size=5, dilation=8, padding=16, bias=False),
            nn.BatchNorm1d(1280),
            nn.GELU(),
            nn.Conv1d(1280, 960, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2),
        )
        self.gelu = nn.GELU()
        self.classifier = finetuneblock1(
            input_channels=960,
            kernel_size=5,
            embed_dim=1280,
            feedforward_dim=2048,
            output_dim=self.n_features,
        )

    def forward(self, input: torch.Tensor):
        output1 = self.lconv_network(input)
        output2 = self.dconv1(output1)
        output3 = self.dconv2(self.gelu(output2 + output1))
        output4 = self.dconv3(self.gelu(output3 + output2))
        output = self.classifier(self.gelu(output3 + output4))
        return output


def build_ConvNet(
    new_model: bool,
    n_features: int,
    model_path: Optional[str] = None,
):
    net = ConvNet(n_features=n_features)
    if not new_model and model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Model succesfully built")
    return net


class AttentionConv(nn.Module):
    def __init__(self, n_features):
        super(AttentionConv, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=480,
                kernel_size=13,
                padding=6,
                bias=False,
            ),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=7),
        )
        self.dropout1 = nn.Dropout(p=0.2)

        self.layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=480,
                out_channels=640,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=640),
            nn.ReLU(),
        )
        self.dropout2 = nn.Dropout(p=0.2)

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=640,
                out_channels=720,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(720),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.dropout3 = nn.Dropout(p=0.2)

        self.attention = nn.MultiheadAttention(embed_dim=720, num_heads=10)
        self.attndrop = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(720)

        self.fc1 = nn.Linear(in_features=720, out_features=2048)
        self.relu1 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.relu2 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=1024, out_features=n_features)

    def forward(self, input):
        output = self.layer1(input)
        output = self.dropout1(output)
        output = self.layer2(output)
        output = self.dropout2(output)
        output = self.layer3(output)
        output = self.dropout3(output)
        output = output.transpose(2, 1)
        att_output = self.attention(query=output, key=output, value=output)[0]
        output = self.norm(self.attndrop(att_output) + output).max(dim=-2)[0]
        # output = output.reshape(output.size(0), -1)
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.dropout4(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout5(output)
        output = self.fc3(output)
        return output


def build_AttenConv(
    new_model: bool,
    n_features: int,
    model_path: Optional[str] = None,
):
    net = AttentionConv(n_features=n_features)
    if not new_model and model_path != None:
        print("Loading model state")
        net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    print("Model succesfully built")
    return net
