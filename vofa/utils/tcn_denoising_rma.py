import torch

class TCNTransposeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = input_tensor.transpose(-1, -2)
        return output_tensor

class TCNRMA(torch.nn.Module):
    def __init__(self, num_act, num_obs, num_obs_stacking,
                 num_privileged_obs, num_critic_priv_obs=0,
                 num_emb=None, num_value=1):
        super().__init__()
        if num_emb is None:
            self.num_emb = 64
        else:
            self.num_emb = num_emb
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs + num_critic_priv_obs, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_value),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs + self.num_emb, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.privileged_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.num_emb),
        )
        self.logstd = torch.nn.parameter.Parameter(
            torch.full((1, num_act,), fill_value=-2.), requires_grad=True)

        linear_size = int((((num_obs_stacking - 3) // 2 - 2) // 2 - 2) // 2 + 1)
        self.adaptation_module = torch.nn.Sequential(
            TCNTransposeLayer(),
            torch.nn.Conv1d(num_obs, 64, 3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(64, 128, 3, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv1d(128, 256, 3, stride=2),
            torch.nn.ELU(),
            TCNTransposeLayer(),
            torch.nn.Flatten(start_dim=-2),
            torch.nn.Linear(linear_size * 256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, self.num_emb),
        )

    def forward(self):
        raise NotImplementedError

    def est_priv_embedding(self, obs_history):
        batch_shape = obs_history.shape[:-2]
        privileged_estimation = self.adaptation_module(
            obs_history.flatten(end_dim=-3)
        ).reshape(*batch_shape, -1)
        return privileged_estimation

    def act(self, obs, privileged=None, obs_history=None, return_emb=False):
        assert privileged is not None or obs_history is not None
        if privileged is not None:
            privileged_embedding = self.privileged_encoder(privileged)
        if obs_history is not None:
            privileged_estimation = self.est_priv_embedding(obs_history)
            if privileged is None:
                privileged_embedding = privileged_estimation
        act_input = torch.cat((obs, privileged_embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        if return_emb:
            ret = [dist,]
            if privileged is not None:
                ret.append(privileged_embedding)
            if obs_history is not None:
                ret.append(privileged_estimation)
            return ret
        else:
            return dist

    def est_value(self, obs, privileged, critic_priv=None):
        if critic_priv is not None:
            critic_input = torch.cat((obs, privileged, critic_priv), dim=-1)
        else:
            critic_input = torch.cat((obs, privileged), dim=-1)
        value = self.critic(critic_input).squeeze(-1)
        return value

    def ac_parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.actor.parameters():
            yield p
        for p in self.privileged_encoder.parameters():
            yield p
        yield self.logstd

    def adapt_parameters(self):
        for p in self.adaptation_module.parameters():
            yield p

class TCNDRMA(TCNRMA):
    def __init__(self, num_act, num_obs, num_obs_stacking,
                 num_privileged_obs, num_critic_priv_obs=0,
                 num_emb=None, num_value=1):
        super().__init__(
            num_act, num_obs, num_obs_stacking,
            num_privileged_obs, num_critic_priv_obs,
            num_emb, num_value)
        self.privileged_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_emb, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, num_privileged_obs),
        )

    def act(self, obs, obs_history, return_priv_est=False):
        priv_embedding = self.est_priv_embedding(obs_history)
        act_input = torch.cat((obs, priv_embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        if return_priv_est:
            priv_est = self.privileged_decoder(priv_embedding)
            return dist, priv_embedding, priv_est
        else:
            return dist

    def adapt_parameters(self):
        for p in self.adaptation_module.parameters():
            yield p
        for p in self.privileged_decoder.parameters():
            yield p


