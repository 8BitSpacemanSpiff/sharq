class CovarianceCollector:
    def __init__(self, layer):
        self.layer = layer
        self.XXT = None
        self.YYT = None
        self.quant_inp = []


    def save_inps(self, _, inp, out):
        self.quant_inp.append(inp[0].data)


    def compute_cov_in_batch(self, _, inp, out):
        if self.XXT is None:
            self.XXT = 0
            self.n_data_in = 0
        
        inp = preprocess(inp[0].data)
        self.XXT, self.n_data_in = compute_cov(self.XXT, self.n_data_in, inp)
            

    def compute_cov_out_batch(self, _, inp, out, n_heads=None):
        if self.YYT is None:
            self.YYT = 0
            self.n_data_out = 0

        out = preprocess(out, n_heads)
        self.YYT, self.n_data_out = compute_cov(self.YYT, self.n_data_out, out)


def preprocess(data, n_heads=None):
    if len(data.shape) == 4:  # [B, H, L, d_h]
        data = data.transpose(0, 1).view(n_heads, -1, data.shape[-1])  # [H, BL, d_h]
    else:
        if len(data.shape) == 2:  # [BL, d]
            data = data.unsqueeze(0)  # [1, BL, d]
        if len(data.shape) == 3:  # [B, L, d]
            data = data.reshape((-1, data.shape[-1]))  # [BL, d]
        if n_heads is not None:
            head_dim = data.shape[-1] // n_heads
            data = data.view(-1, n_heads, head_dim).transpose(0, 1).contiguous()  # [H, BL, d_h]
    data = data.transpose(-1, -2)  # [d, BL] or [H, d_h, BL]

    return data


def compute_cov(cov, n_data, data_1, data_2=None):
    if data_2 is None:
        data_2 = data_1
    n_data_new = data_1.shape[-1]
    cov *= n_data / (n_data + n_data_new)
    n_data += n_data_new
    cov += (2 / n_data) * (data_1.float() @ data_2.float().transpose(-1, -2))

    return cov, n_data


