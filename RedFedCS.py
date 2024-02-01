
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import fr_aggregation

class RepFedCS(ClientSelection):
    def __init__(self, device, select_ratio=0.1):
        super().__init__()
        '''
        Args:
            select_ratio: 0.1
        '''
        if select_ratio is None:
            sys.exit('Please set the hyperparameter: subset ratio! =)')
        self.select_ratio = select_ratio

    def select(self,n, client_idxs, protos):
        global_fr = fr_aggregation(protos)
        # n = int(len(protos)*self.subset_ratio)
        frs_w = self.kd(global_fr,frs)
        # print(proto_w)
        frs_we = np.take(proto_w.numpy(), client_idxs) 
        selected_clients = np.random.choice(client_idxs, n, p=frs_we/sum(frs_we), replace=False)
        return selected_clients

    def kd(self, global_frs, frs):
        with torch.no_grad():
            client_global_distance=torch.zeros(len(protos),1)
            for client_id, fr_v in frs.items():
                _distance=0
                for proto_id, fr in fr_v.items():
                    _distance += F.cosine_similarity(fr, global_frs[fr_id][0],dim=0)
                client_global_distance[client_id]=_distance
            # print(_distance)
            client_global_distance = torch.nan_to_num(client_global_distance)
            fr_w = client_global_distance / torch.sum(client_global_distance)
            # print(torch.sum(client_global_distance))
        return fr_w
