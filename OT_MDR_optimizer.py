import torch
import random
import time


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, is_sgd=False, model=None,
                 ignore_sigma=False, geometry=False, p_power=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.is_sgd = is_sgd
        self.ignore_sigma = ignore_sigma
        self.geometry = geometry
        self.p_power = p_power
        self.model = model
        if self.model: self.assign_name(self.model)
        self.count = 0

    def assign_name(self, model):
        self.param_by_name = {}
        for n, p in model.named_parameters():
            self.state[p]["name"] = n
            self.param_by_name[n] = p

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.count += 1
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                self.state[p]["old_p"] = p.data.clone()
                if p.grad is None: continue
                self.state[p]["old_p_grad"] = p.grad.data.clone()
                if self.model:
                    name_p = self.state[p]['name']
                    if self.ignore_sigma and "rho" in name_p:
                        continue
                s = (p if group["adaptive"] else torch.tensor(1.0).to(p))
                if self.geometry:
                    s = torch.abs(p)
                    if self.model:
                        if "rho" in name_p:
                            if self.ignore_sigma:
                                continue
                            else:
                                name_p_mu = name_p.replace("rho", "mu")
                                p_mu = self.param_by_name[name_p_mu]
                                # sigma = 1.0
                                # if self.count > 10:
                                #     sigma = torch.max(torch.log1p(torch.exp(p)), torch.tensor(1e-6).to(p))
                                # s = p_mu / sigma
                                # same like in the paper
                                sigma = p
                                if self.count < 10:
                                    sigma = 1.0
                                s = torch.abs(p_mu) / sigma
                        elif "mu" in name_p:
                            name_p_rho = name_p.replace("mu", "rho")
                            p_rho = self.param_by_name[name_p_rho]
                            # if self.p_power:
                            #     s = p / torch.log1p(torch.exp(p_rho**2))
                            # else:
                            #     sigma = 1.0
                            #     if self.count > 10:
                            #         sigma = torch.max(torch.log1p(torch.exp(p_rho)), torch.tensor(1e-6).to(p))
                            #     s = p / sigma
                            # same like in the paper
                            sigma = p_rho
                            if self.count < 10:
                                sigma = 1.0
                            s = torch.abs(p.data) / sigma
                # e_w =  * p.grad * scale.to(p)
                e_w = torch.pow(s, 2) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
                if p.grad is None: continue
                if self.is_sgd:
                    p.grad.data = self.state[p]["old_p_grad"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self, geometry=None, adaptive=None):
        if geometry is None:
            geometry = self.geometry
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if geometry:
            norm = []
            for group in self.param_groups:
                for p in group["params"]:
                    scale = torch.abs(p)
                    if self.model:
                        name_p = self.state[p]['name']
                        if "rho" in name_p:
                            if self.ignore_sigma: continue
                            else:
                                name_p_mu = name_p.replace("rho", "mu")
                                p_mu = self.param_by_name[name_p_mu]
                                # sigma = torch.max(torch.log1p(torch.exp(p)), torch.tensor(1e-6).to(shared_device))
                                # if self.count < 10:
                                #     sigma = 1.0
                                # scale = torch.abs(p_mu) / sigma
                                # same like in the paper
                                sigma = p
                                if self.count < 10:
                                    sigma = 1.0
                                scale = torch.abs(p_mu) / sigma

                            sigma = p
                            if self.count < 10:
                                sigma = 1.0
                            scale = torch.abs(p_mu) / sigma
                        elif "mu" in name_p:
                            name_p_rho = name_p.replace("mu", "rho")
                            p_rho = self.param_by_name[name_p_rho]
                            # if self.p_power:
                            #     sigma = 1.0
                            #     if self.count > 10:
                            #         sigma = torch.max(torch.log1p(torch.exp(p_rho)), torch.tensor(1e-6).to(shared_device))
                            #     scale = torch.abs(p) / sigma
                            # else:
                            #     sigma = 1.0
                            #     if self.count > 10:
                            #         sigma = torch.max(torch.log1p(torch.exp(p_rho)), torch.tensor(1e-6).to(shared_device))
                            #     scale = torch.abs(p.data) / sigma
                            # # same like in the paper
                            sigma = p_rho
                            if self.count < 10:
                                sigma = 1.0
                            scale = torch.abs(p.data) / sigma
                    if p.grad is not None:
                        norm.append((scale*p.grad).norm(p=2).to(shared_device))
            norm = torch.norm(torch.stack(norm), p=2)
        else:
            norm = []
            for group in self.param_groups:

                is_adaptive = adaptive
                if is_adaptive is None:
                    is_adaptive = group["adaptive"]

                for p in group["params"]:
                    if p.grad is None: continue
                    if self.model:
                        name_p = self.state[p]['name']
                        if "rho" in name_p and self.ignore_sigma:
                            continue
                    norm.append(((torch.abs(p) if is_adaptive else 1.0) * p.grad).norm(p=2).to(shared_device))
            norm = torch.norm(torch.stack(norm), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAM_batch_chain(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, perturb_alpha=0.5, rho_lst=[0.05, 0.05], noise_var=0.0,
                 merge_grad=False, mode=1, model=None, ignore_sigma=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, rho1=rho_lst[0], rho2=rho_lst[1], adaptive=adaptive, **kwargs)
        super(SAM_batch_chain, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.perturb_eps = 1e-12
        self.rms_warmup = 100
        self.perturb_alpha = perturb_alpha
        self.count = [0]*20
        self.count_dic = {}
        self.num_iter = 0
        self.num_call_1st = 0
        self.noise_var = noise_var
        self.merge_grad = merge_grad
        self.mode = mode
        self.ignore_sigma = ignore_sigma
        self.model = model
        if self.model: self.assign_name(self.model)

    def assign_name(self, model):
        self.param_by_name = {}
        for n, p in model.named_parameters():
            self.state[p]["name"] = n
            self.param_by_name[n] = p

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:

            rho = group["rho{}".format(self.num_call_1st+1)]
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                if self.num_call_1st == 0:
                    self.state[p]["old_p"] = p.data.clone()
                if self.model:
                    name_p = self.state[p]['name']
                    if self.ignore_sigma and "rho" in name_p:
                        continue
                p_grad = p.grad.data.clone()
                if self.merge_grad:
                    for i in range(self.num_call_1st):
                        p_grad += self.state[p]["p_grad_{}".format(i)]

                    if self.mode == 1.1:
                        p_grad = p_grad/(self.num_call_1st+1)
                self.state[p]["p_grad_{}".format(self.num_call_1st)] = p_grad
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p_grad * scale.to(p)
                noise = torch.randn_like(p_grad)*self.noise_var
                p.add_(e_w+noise)  # climb to the local maximum "w + e(w)"  <- theta_c = theta_max
        self.num_call_1st += 1
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False,  mode=1.0, noise_var=0):

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        self.num_call_1st = 0
        if zero_grad: self.zero_grad()

    def _grad_norm(self, by=None):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device

        def check(p):
            if p.grad is None:
                return False
            # if self.ignore_sigma and "rho" in self.state[p]['name']:
            #     return False
            return True

        if not by:
            norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p.data) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if check(p)
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if group["adaptive"] else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if check(p)
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAMnChain(torch.optim.Optimizer):
    """
    A replica of our beloved SAM_batch_chain but with n particles
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, perturb_alpha=0.5, rho_lst=[0.1, 0.2], noise_var=0,
                 merge_grad=False, mode=1, n_branch=2, true_grad=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho1=rho_lst[0], rho2=rho_lst[1], adaptive=adaptive, **kwargs)
        super(SAMnChain, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.perturb_eps = 1e-12
        self.rho_lst = rho_lst
        # self.rms_rho = [0.7, 0.7]
        self.rms_warmup = 100
        self.perturb_alpha = perturb_alpha
        self.count = [0]*20
        self.count_dic = {}
        self.num_iter = 0
        self.num_call_1st = 0
        self.noise_var = noise_var
        self.merge_grad = merge_grad
        self.mode = mode
        self.idx_part = 0
        self.grad_norm = {}
        self.count_2nd = 0
        self.true_grad = true_grad
        self.n_branch = n_branch


    @torch.no_grad()
    def first_step(self, current_part, zero_grad=False, p_a=True):
        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None: continue

                self.state[p]["old_p"] = p.data.clone()
                p_grad = p.grad.data.clone()
                self.state[p]["{}_p_grad_{}".format(current_part, 1)] = p_grad.clone()
        if p_a:
            by = "{}_p_grad_{}".format(current_part, 1)
            grad_norm = self._grad_norm(by=by)
            for group in self.param_groups:
                rho = group["rho1"]
                scale = rho / (grad_norm + self.perturb_eps)
                for p in group["params"]:
                    if p.grad is None: continue

                    # p.data = self.state[p]["old_p"].clone()
                    p_grad = self.state[p]["{}_p_grad_{}".format(current_part, 1)].clone()

                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p_grad * scale.to(p)
                    noise = torch.randn_like(p_grad) * self.noise_var
                    p.add_(e_w + noise)  # p_tiu
                    self.state[p]["{}_p_tiu".format(current_part)] = p.data.clone()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def first_step_grad(self, current_part, zero_grad=False, p_a=True):
        # grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:

                if p.grad is None: continue

                p_grad = p.grad.data.clone()
                self.state[p]["{}_p_grad_{}".format(current_part, 2)] = p_grad.clone()
                if self.merge_grad:
                    p_grad += self.state[p]["{}_p_grad_{}".format(current_part, 1)]
                self.state[p]["{}_p_grad_{}_merge".format(current_part, 2)] = p_grad.clone()
        if p_a:
            by = "{}_p_grad_{}".format(current_part, 2)
            if self.true_grad:
                by = "{}_p_grad_{}_merge".format(current_part, 2)
            grad_norm = self._grad_norm(by=by)
            for group in self.param_groups:
                rho = group["rho2"]
                scale = rho / (grad_norm + self.perturb_eps)
                for p in group["params"]:
                    if p.grad is None: continue
                    p_grad = self.state[p]["{}_p_grad_{}_merge".format(current_part, 2)].clone()

                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p_grad * scale.to(p)
                    noise = torch.randn_like(p_grad) * group['lr'] * self.noise_var
                    p.add_(e_w + noise)  # climb to the local maximum "w + e(w)"  <- theta_c = theta_max
                    self.state[p]["{}_p_a".format(current_part)] = p.data.clone()
        if zero_grad: self.zero_grad()

    def save_grad(self, current_part, name="p_grad_update", data_name=None, zero_grad=False):
        # import pdb; pdb.set_trace()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["{}_{}".format(current_part, name)] = p.grad.data.clone()
                if data_name:
                    p.data = self.state[p][data_name].clone()  # "old_p"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False,  mode=1.0, noise_var=0):
        # grad_particles = []
        # for _ in range(self.n_branch):
        #     grad_particles.append([])
        # countG = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data = self.state[p]["old_p"].clone()
                if "{}_p_grad_update".format(0) not in self.state[p]: continue
                grad = []
                for idx_p in range(self.n_branch):
                    # grad_particles[idx_p].append(self.state[p]["{}_p_grad_update".format(idx_p)].clone())
                    grad.append(self.state[p]["{}_p_grad_update".format(idx_p)].clone())
                # p.grad.data = sum(grad)/len(grad)
                if self.mode == 1:
                    p.grad = sum(grad)
                elif self.mode == 1.1:
                    p.grad = sum(grad) / len(grad)
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self, by=None):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p.data) if group["adaptive"] else 1.0) * p.grad.data).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if group["adaptive"] else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


