import torch

class AsymmetricActorCritic(torch.nn.Module):
    def __init__(self, num_act, num_prop, hist_len, num_privileged_obs, init_logstd=-2.0):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_prop + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_prop, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=init_logstd), requires_grad=False)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    @torch.jit.export
    def forward(self, obs):
        out = obs[..., -1, :]
        action_mean = self.actor(out)
        action_mean = torch.tanh(action_mean) # squash to [-1, 1]
        return action_mean
    
    def act(self, obs, privileged_obs):
        '''
        obs: (batch_size, hist_len, num_prop) / (horizon_length, num_envs, hist_len, num_prop)
        privileged_obs: (batch_size, num_privileged_obs) / (horizon_length, num_envs, num_privileged_obs)
        '''
        out = obs[..., -1, :]
        action_mean = self.actor(out)
        action_mean = torch.tanh(action_mean) # squash to [-1, 1]
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)
    
    def est_value(self, obs, privileged_obs):
        '''
        obs: (batch_size, hist_len, num_prop) / (horizon_length, num_envs, hist_len, num_prop)
        privileged_obs: (batch_size, num_privileged_obs) / (horizon_length, num_envs, num_privileged_obs)
        '''
        critic_input = torch.cat((obs[..., -1, :], privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)