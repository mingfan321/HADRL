import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F


class StateMCSRP(NamedTuple):
    # Fixed input
    p_size: torch.Tensor
    coords: torch.Tensor  # Depot + loc
    distin: torch.Tensor
    # Max length is not a single value, but one for each node indicating max length tour should have when arriving
    # at this node, so this is max_length - d(depot, node)
    max_length: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    cum_lengths: torch.Tensor  # 累积长度
    cur_lengths: torch.Tensor   # 当前长度
    cur_coord: torch.Tensor
    distances: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            cum_lengths=self.cum_lengths[key],
            cur_lengths=self.cur_lengths[key],
            cur_coord=self.cur_coord[key],
            # cur_total_prize=self.cur_total_prize[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        p_size = int(input['p_size'][-1])
        loc = input['loc']
        distin = input['distin']
        max_length = input['max_length']

        batch_size, n_loc, _ = loc.size()

        return StateMCSRP(
            p_size=p_size,
            coords=loc,
            distin=distin,
            # max_length is max length allowed when arriving at node, so subtract distance to return to depot
            # Additionally, substract epsilon margin for numeric stability
            max_length=max_length[:, None],
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            cum_lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=loc[:, 0][:, None, :],  # Add step dimension
            # distance matrix from target nodes to charging nodes
            distances=(loc[:, p_size:, :].repeat((1, 1, p_size)).reshape((-1, n_loc-p_size, p_size, 2)) - loc[:, None, :p_size, :]).norm(p=2, dim=-1) + 1e-6,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_remaining_length(self):
        # max_length[:, 0] is max length arriving at depot so original max_length
        return self.max_length[self.ids, 0] - self.cur_lengths

    def get_final_cost(self):

        assert self.all_finished()
        # The cost is the negative of the collected prize since we want to maximize collected prize
        return self.cum_lengths

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        # print(prev_a)

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        cum_lengths = self.cum_lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        segment_length = self.cur_lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)
        assert (segment_length <= self.max_length).all(), print(prev_a[segment_length > self.max_length])
        cur_lengths = segment_length*(torch.ge(prev_a, self.p_size)).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            cum_lengths=cum_lengths, cur_lengths=cur_lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        # return self.i.item() > 0 and (self.prev_a == 0).all()
        # return self.visited_[:, :, self.p_size:].all() and (self.prev_a == 0).all()
        # print((self.prev_a == 0).all(), self.visited_[:, :, self.p_size:].all())
        return (self.prev_a == 0).all() and self.visited_[:, :, self.p_size:].all()

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):

        # print(self.prev_a)
        rlength = self.max_length[self.ids, :] - self.cur_lengths[:, :, None]   # remain flight length
        b = (self.coords[self.ids, self.p_size:, :] - self.cur_coord[:, :, None, :]).norm(p=2, dim=-1)
        c = (rlength - b).squeeze(-2)
        d = c[:, :, None].repeat((1, 1, self.p_size-1)) < self.distances[self.ids, :, 1:].squeeze(1)
        exceeds_length = torch.ge(d.sum(-1).unsqueeze(1), self.p_size-1).to(d.dtype)
        mask_loc = self.visited_[:, :, self.p_size:].to(d.dtype) | exceeds_length

        exceeds_depot = rlength.repeat((1, 1, self.p_size)) < (self.coords[self.ids, :self.p_size, :] - self.cur_coord[:, :, None, :]).norm(p=2, dim=-1)

        exceeds_depot[:, :, 1:] = exceeds_depot[:, :, 1:] | (self.prev_a < self.p_size)[:, :, None].expand([mask_loc.size(0), 1, self.p_size -1])
        # mask depot if there exists unvisited target node
        exceeds_depot[:, :, 0] = exceeds_depot[:, :, 0] | ((self.visited_[:, :, self.p_size:] == 0).int().sum(-1) > 0)
        mask_all = torch.cat((exceeds_depot, mask_loc), -1)
        assert not (mask_all.sum(-1) == mask_all.size(-1)).any(), print(rlength[exceeds_depot.sum(-1) == 2], )

        return mask_all

    def construct_solutions(self, actions):
        return actions
