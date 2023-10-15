import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


def clean_cache():
    cur_gpu = torch.cuda.current_device()
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu)
        torch.cuda.empty_cache()
    torch.cuda.set_device(cur_gpu)
    return


def head_diff(x, head_idx, head_length):
    result = torch.empty(x.shape[0], x.shape[1], 0, x.shape[3], device=x.device, dtype=x.dtype)
    k=0
    head_length = head_length // 3
    uplimit = (x.shape[2] // head_length) - 3
    while k < uplimit:
        element = x[:,:,head_idx+k*head_length,...].unsqueeze(2)
        result = torch.cat((result, element), dim=2)
        k += 1
    return result


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]


    def get_method(self, method_str):
        METHOD_DICT = {
            "NotCut": self.not_cut,
            "StreamLLM": self.evict_for_space,
            "SparseSample": self.diff_head
        }
        return METHOD_DICT[method_str]
    
    @staticmethod
    def get_supported_method():
        return ["NotCut", "StreamLLM", "SparseSample"]

    def not_cut(self, past_key_values, space_needed):
        return past_key_values

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        # slice start window and recent window for k,v, then concat them
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    # Use by streamLLM
    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        # [(Samples),(K or V),(data)]
        cache = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

        del past_key_values
        clean_cache()

        return cache
    
    # Sparse Sample by us
    def diff_head(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)

        # if seq_len + num_coming <= self.cache_size:
            # return past_key_values
        # [(Samples),(K or V),(data)]
        cache = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        head_diff(
                            k, head_idx, len(past_key_values)
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        head_diff(
                            v, head_idx, len(past_key_values)
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for head_idx,(k, v) in enumerate(past_key_values)
        ]

        ##print(f"({len(cache)},{len(cache[0])},{cache[0][0].shape})")
        #input()
        del past_key_values
        clean_cache()
        
        return cache

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
