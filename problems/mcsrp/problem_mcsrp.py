from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.mcsrp.state_mcsrp import StateMCSRP
from utils.beam_search import beam_search


class MCSRP(object):
    NAME = 'mcsrp'  # Multiple Charging Sation Routing Problem

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        # assert (
        #         torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
        #         pi.data.sort(1)[0]
        # ).all(), "Invalid tour"

        # Gather dataset in order of tour
        loc = dataset['loc']
        d = loc.gather(1, pi[..., None].expand(*pi.size(), loc.size(-1)))
        # Check that tours are valid
        rolled_d = d.roll(dims=1, shifts=1)
        lengths = ((d-rolled_d)**2).sum(2).sqrt()
        cum_length = torch.cumsum(lengths, dim=1)
        idx = (pi >= dataset['p_size'][:, None])
        cum_length[idx] = 0
        sorted_cum_length, _ = cum_length.sort(axis=1)
        rolled_sorted_cum_length = sorted_cum_length.roll(dims=1, shifts=1)
        diff_mat = sorted_cum_length - rolled_sorted_cum_length
        diff_mat[diff_mat < 0] = 0
        makespans, _ = torch.max(diff_mat, dim=1)

        assert (makespans <= dataset['max_length']).all(), print(makespans[makespans > dataset['max_length']])
        # cost = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - loc[:, 0]).norm(p=2, dim=1)

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return cum_length[:, -1], None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return MCSRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMCSRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = MCSRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, c_size):
    # size: the number of targets
    # p_size: the number of the charging stations including the depot

    MAX_LENGTHS = {
        20: 3.,
        50: 3.,
        100: 3.
    }

    loc_t = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)
    station = torch.randint(0, 5, (c_size-1, 2)) / 4
    # loc[0] = torch.FloatTensor([0.5, 0.5])  # determine the center as the depot
    # distinguish the locations of the charging station and target_points
    loc = torch.cat((depot[None, :], station, loc_t))
    distin = torch.cat((torch.ones(c_size, dtype=torch.int64), torch.zeros(size, dtype=torch.int64)))
    distin[0] = torch.zeros(1, dtype=torch.int64)  # distinguish depot

    return {
        'p_size': torch.tensor(c_size),
        'loc': loc,
        # Uniform 1 - 9, scaled by capacities
        'distin': distin,
        'max_length': torch.tensor(MAX_LENGTHS[size])
    }


class MCSRPDataset(Dataset):

    def __init__(self, filename=None, size=50, p_size=4, num_samples=1000000, offset=0):
        super(MCSRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'p_size': torch.tensor(p_size),
                        'loc': torch.FloatTensor(loc),
                        'distin': torch.tensor(distin),
                        'max_length': torch.tensor(max_length)
                    }
                    for p_size, loc, distin, max_length in (data[offset:offset + num_samples])
                ]
        else:
            self.data = [
                generate_instance(size, p_size)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
